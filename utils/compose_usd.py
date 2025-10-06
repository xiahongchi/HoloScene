from plyfile import PlyData
import omni
from isaacsim import SimulationApp
import numpy as np
from utils.general import detect_collision
import trimesh
simulation_app = None

def start_simulation_app():

    global simulation_app
    simulation_app = SimulationApp({"headless": True})
    print("Starting simulation app...")
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils



# Step 2: Get object according to given USD prim path
def get_prim(prim_path):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim at path {prim_path} is not valid.")
        return None
    return prim


# Helper function to extract position and orientation from the transformation matrix
def extract_position_orientation(transform):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    position = Gf.Vec3d(transform.ExtractTranslation())
    rotation = transform.ExtractRotationQuat()
    orientation = Gf.Quatd(rotation.GetReal(), *rotation.GetImaginary())
    return position, orientation


def quaternion_angle(q1, q2):
    """
    Calculate the angle between two quaternions.

    Parameters:
    q1, q2: Lists or arrays of shape [w, x, y, z] representing quaternions

    Returns:
    angle: The angle in radians between the two quaternions
    """
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    # Convert lists to numpy arrays if they aren't already
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate the relative quaternion: q_rel = q2 * q1^(-1)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])  # Inverse of a normalized quaternion

    # Quaternion multiplication for q_rel = q2 * q1_inv
    q_rel = np.array([
        q2[0] * q1_inv[0] - q2[1] * q1_inv[1] - q2[2] * q1_inv[2] - q2[3] * q1_inv[3],
        q2[0] * q1_inv[1] + q2[1] * q1_inv[0] + q2[2] * q1_inv[3] - q2[3] * q1_inv[2],
        q2[0] * q1_inv[2] - q2[1] * q1_inv[3] + q2[2] * q1_inv[0] + q2[3] * q1_inv[1],
        q2[0] * q1_inv[3] + q2[1] * q1_inv[2] - q2[2] * q1_inv[1] + q2[3] * q1_inv[0]
    ])

    # The angle can be calculated from the scalar part (real part) of the relative quaternion
    angle = 2 * np.arccos(min(abs(q_rel[0]), 1.0))

    return angle * 180 / np.pi  # Convert to degrees


# Step 3: Start simulation and trace position, orientation, and speed of the object
def start_simulation_and_trace(prims, duration=1.0, dt=1.0 / 60.0):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    global simulation_app
    # Define a list to store the traced data
    traced_data = {}
    init_data = {}

    # Get the timeline and start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Initialize variables for tracking the previous position for speed calculation
    prev_position = None
    elapsed_time = 0.0
    init = True

    while elapsed_time < duration:

        # Get the current time code
        current_time_code = Usd.TimeCode.Default()

        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)

        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
            # Calculate speed if previous position is available

            # Store the data for current frame
            traced_data_prim = traced_data.get(f"{prim_i}", [])

            if init:
                init_data[f"{prim_i}"] = {}
                init_data[f"{prim_i}"]["position"] = [position[0], position[1], position[2]]
                init_data[f"{prim_i}"]["orientation"] = [orientation.GetReal(),
                                                         orientation.GetImaginary()[0],
                                                         orientation.GetImaginary()[1],
                                                         orientation.GetImaginary()[2]
                                                         ]
                relative_position = np.array([0, 0, 0])
                relative_orientation = 0.

            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[f"{prim_i}"]["position"][0],
                                          init_data[f"{prim_i}"]["position"][1],
                                          init_data[f"{prim_i}"]["position"][2]])

                orientation_cur = np.array([orientation.GetReal(),
                                            orientation.GetImaginary()[0],
                                            orientation.GetImaginary()[1],
                                            orientation.GetImaginary()[2]
                                            ])
                orientation_init = np.array([init_data[f"{prim_i}"]["orientation"][0],
                                             init_data[f"{prim_i}"]["orientation"][1],
                                             init_data[f"{prim_i}"]["orientation"][2],
                                             init_data[f"{prim_i}"]["orientation"][3]
                                             ])

                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)

            relative_position = relative_position.tolist()
            relative_orientation = float(relative_orientation)

            traced_data_prim.append({
                "time": elapsed_time,
                "position": relative_position,
                "orientation": relative_orientation,
            })

            traced_data[f"{prim_i}"] = traced_data_prim

        if init:
            init = False
        print(f"\relapsed_time: {elapsed_time:.5f}", end="")
        # Step the simulation
        simulation_app.update()
        # Increment the elapsed time
        elapsed_time += dt

    # Stop the simulation
    timeline.stop()

    return traced_data


def AddTilt(top):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    tilt = top.AddRotateXOp(opSuffix='tilt')
    tilt.Set(value=90)


def AddTranslate(top, offset):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    top.AddTranslateOp().Set(value=offset)


def convert_ply_to_usd(stage, usd_internal_path, ply_file_path, collision_approximation, static):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    plydata = PlyData.read(ply_file_path)
    n_verts = len(plydata.elements[0].data)
    n_faces = len(plydata.elements[1].data)

    points = []
    point_colors = []
    for vert_i in range(n_verts):
        points.append(np.array(list(plydata.elements[0].data[vert_i])).reshape(-1)[:3])
        point_colors.append(np.array(list(plydata.elements[0].data[vert_i])).reshape(-1)[3:6] / 255.)
    points = np.stack(points, axis=0)

    bbox_max = np.max(points, axis=0)
    bbox_min = np.min(points, axis=0)
    center = (bbox_max + bbox_min) / 2
    points = points - center
    center = (center[0], center[1], center[2])

    point_colors = np.stack(point_colors, axis=0)

    faces = []
    vertex_counts = []

    for face_i in range(n_faces):
        faces.append(np.array(plydata.elements[1].data[face_i][0].tolist()).reshape(-1).astype(np.int32))
        vertex_counts.append(3)

    faces = np.concatenate(faces, axis=0)
    vertex_counts = np.array(vertex_counts).astype(np.int32)

    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(point_colors))
    mesh.CreateDisplayColorPrimvar("vertex")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    tilt = mesh.AddRotateXOp(opSuffix='tilt')
    tilt.Set(value=-90)
    AddTranslate(mesh, center)

    prim = stage.GetPrimAtPath(usd_internal_path)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        ps_rigid_api.CreateSolverPositionIterationCountAttr(255)
        ps_rigid_api.CreateSolverVelocityIterationCountAttr(255)
        ps_rigid_api.CreateEnableCCDAttr(True)
        ps_rigid_api.CreateEnableSpeculativeCCDAttr(True)

    UsdPhysics.CollisionAPI.Apply(prim)
    ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    ps_collision_api.CreateContactOffsetAttr(0.4)
    ps_collision_api.CreateRestOffsetAttr(0.)
    # collider.CreateApproximationAttr("convexDecomposition")
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rigid_body.CreateLinearDampingAttr(50.0)
    physx_rigid_body.CreateAngularDampingAttr(200.0)

    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

    if collision_approximation == "sdf":
        physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        physx_sdf.CreateSdfResolutionAttr(256)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("sdf")
    elif collision_approximation == "convexDecomposition":
        convexdecomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("convexDecomposition")

    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateDynamicFrictionAttr(1e20)
    mat.CreateStaticFrictionAttr(1e20)

    return stage


def convert_mesh_to_usd(stage, usd_internal_path, verts, faces, collision_approximation, static):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]

    points = verts

    bbox_max = np.max(points, axis=0)
    bbox_min = np.min(points, axis=0)
    center = (bbox_max + bbox_min) / 2
    points = points - center
    center = (center[0], center[1], center[2])

    vertex_counts = np.ones(n_faces).astype(np.int32) * 3

    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    # mesh.CreateDisplayColorPrimvar("vertex")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateExtentAttr([(-10, -10, -10), (10, 10, 10)])

    tilt = mesh.AddRotateXOp(opSuffix='tilt')
    tilt.Set(value=-90)
    AddTranslate(mesh, center)

    prim = stage.GetPrimAtPath(usd_internal_path)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        ps_rigid_api.CreateSolverPositionIterationCountAttr(255)
        ps_rigid_api.CreateSolverVelocityIterationCountAttr(255)
        ps_rigid_api.CreateEnableCCDAttr(True)
        ps_rigid_api.CreateEnableSpeculativeCCDAttr(True)

    UsdPhysics.CollisionAPI.Apply(prim)
    ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    ps_collision_api.CreateContactOffsetAttr(0.4)
    ps_collision_api.CreateRestOffsetAttr(0.)
    # collider.CreateApproximationAttr("convexDecomposition")
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rigid_body.CreateLinearDampingAttr(50.0)
    physx_rigid_body.CreateAngularDampingAttr(200.0)

    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

    if collision_approximation == "sdf":
        physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        physx_sdf.CreateSdfResolutionAttr(256)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("sdf")
    elif collision_approximation == "convexDecomposition":
        convexdecomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("convexDecomposition")

    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateDynamicFrictionAttr(1e20)
    mat.CreateStaticFrictionAttr(1e20)

    return stage


def compose_usd_file_from_mesh(mesh_paths, static_list):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    stage = Usd.Stage.CreateInMemory()

    collision_approximation = "sdf"

    for mesh_i, source_mesh_path in enumerate(mesh_paths):
        usd_internal_path = f"/obj_{mesh_i}"
        usd_internal_path = usd_internal_path.replace("-", "_")
        stage = convert_ply_to_usd(stage, usd_internal_path, source_mesh_path, collision_approximation,
                                   static_list[mesh_i])

    return stage


def start_sim(stage, total_obj_num, movable_id):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    if stage is None:
        assert False, "Failed to load USD stage."

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    omni.usd.get_context().attach_stage_with_callback(stage_id)
    prims = []

    # Get the prim of the object
    for prim_i in range(total_obj_num):
        usd_prim_path = f"/obj_{prim_i}"
        prim = get_prim(usd_prim_path)
        if prim is None:
            assert False, f"Failed to get prim at path {usd_prim_path}"
        prims.append(prim)

    # Start the simulation and trace the object
    traced_data = start_simulation_and_trace(prims)
    relative_orientation = traced_data[f"{movable_id}"][-1]["orientation"]

    omni.usd.get_context().close_stage()
    return relative_orientation


def compose_usd_file_from_meshes_detect_collision(all_meshes, movable_id):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    final_translation = np.array([0, 0, 0])

    # calculate the average edge length of    all_meshes[movable_id]
    edge_length = np.mean(np.linalg.norm(all_meshes[movable_id].vertices[all_meshes[movable_id].faces[:, 0]] -
                                         all_meshes[movable_id].vertices[all_meshes[movable_id].faces[:, 1]], axis=1))
    for _ in range(100):
        contact_points, contact_mesh_id, contact_face_id, contact_face_normals = \
            detect_collision([(mesh.vertices, mesh.faces, mesh.face_normals) for i, mesh in enumerate(all_meshes) if
                              i != movable_id],
                             (all_meshes[movable_id].vertices, all_meshes[movable_id].faces))

        if len(contact_points) == 0:
            break

        average_normals = np.mean(contact_face_normals, axis=0)
        average_normals = average_normals / np.linalg.norm(average_normals)

        average_normals = average_normals.reshape(1, 3)

        all_meshes[movable_id].vertices = all_meshes[movable_id].vertices + average_normals * edge_length
        final_translation = final_translation + average_normals * edge_length

    # import open3d as o3d
    # # create a point cloud
    # pcd = o3d.geometry.PointCloud()
    # # use contact_points as the points
    # pcd.points = o3d.utility.Vector3dVector(contact_points)
    # o3d.io.write_point_cloud('./exps/phyrecon_grid_replica_room_0_gt_mask/2025_04_15_17_38_20/plots/surface_100_2_contact.ply', pcd)
    # assert False

    # print("final_translation: ", final_translation)
    # trimesh.exchange.export.export_mesh(
    #     all_meshes[movable_id],
    #     './exps/phyrecon_grid_replica_room_0_gt_mask/2025_04_15_17_38_20/plots/surface_100_2_relocate.ply',
    # )
    # assert False

    static_list = [True] * len(all_meshes)
    static_list[movable_id] = False

    stage = Usd.Stage.CreateInMemory()

    collision_approximation = "sdf"

    for mesh_i in range(len(all_meshes)):
        usd_internal_path = f"/obj_{mesh_i}"
        usd_internal_path = usd_internal_path.replace("-", "_")
        stage = convert_mesh_to_usd(stage, usd_internal_path,
                                    all_meshes[mesh_i].vertices, all_meshes[mesh_i].faces,
                                    collision_approximation, static_list[mesh_i])

    return stage


def compose_usd_file_from_mesh_paths_detect_collision(mesh_paths, parent_id, movable_id):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    all_meshes = [trimesh.load(mesh_path) for mesh_path in mesh_paths]

    stage = compose_usd_file_from_meshes_detect_collision(all_meshes, parent_id, movable_id)
    return stage


def sim_validation(mesh_list):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils

    mesh_list_copy = [trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy()) for mesh in mesh_list]
    movable_id = len(mesh_list_copy) - 1
    stage = compose_usd_file_from_meshes_detect_collision(mesh_list_copy, movable_id)
    relative_orientation = start_sim(stage, len(mesh_list_copy), movable_id)
    return relative_orientation

