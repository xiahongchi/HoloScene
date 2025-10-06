import numpy as np
import base64
from PIL import Image
from io import BytesIO
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, 
    Buffer, BufferView, Accessor, 
    Image as GLTFImage, Texture, Sampler, Material, PbrMetallicRoughness,
    FLOAT, UNSIGNED_INT, SCALAR, VEC2, VEC3, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
)
import argparse
import sys
import trimesh
sys.path.append('.')
sys.path.append('./MVMeshRecon')
import os
from pyhocon import ConfigFactory
import glob
from pytorch3d.io import save_obj, load_objs_as_meshes
from utils.general import load_tex_dict_from_tex_mesh_p3d, solve_intersection
import pickle

_current_scene = None

def create_glb_scene():
    global _current_scene
    gltf = GLTF2()
    gltf.asset = {"version": "2.0"}
    gltf.scenes = []
    gltf.nodes = []
    gltf.meshes = []
    gltf.materials = []
    gltf.textures = []
    gltf.images = []
    gltf.samplers = []
    gltf.buffers = []
    gltf.bufferViews = []
    gltf.accessors = []
    
    scene = Scene(nodes=[])
    gltf.scenes.append(scene)
    gltf.scene = 0
    
    _current_scene = gltf
    return gltf

def add_textured_mesh_to_glb_scene(textured_mesh_dict, scene=None, material_name="Material", mesh_name="Mesh", preserve_coordinate_system=True):
    global _current_scene
    
    if scene is None:
        scene = _current_scene
    
    if scene is None:
        raise ValueError("No scene available. Call create_glb_scene() first.")
    
    vertices = textured_mesh_dict['vertices'].astype(np.float32)
    faces = textured_mesh_dict['faces'].astype(np.uint32)
    vt = textured_mesh_dict['vt'].astype(np.float32)
    # vt[:, 1] = 1.0 - vt[:, 1]
    ft = textured_mesh_dict['ft'].astype(np.uint32)
    texture_map = textured_mesh_dict['texture_map']
    
    if preserve_coordinate_system:
        vertices_transformed = vertices.copy()
        vertices_transformed[:, [1, 2]] = vertices[:, [2, 1]]
        vertices_transformed[:, 2] = -vertices_transformed[:, 2]
        vertices = vertices_transformed
    
    if texture_map.dtype != np.uint8:
        texture_map = (texture_map * 255).astype(np.uint8)
    
    if faces.max() >= len(vertices):
        raise ValueError(f"Face indices exceed vertex count: max face index {faces.max()}, vertex count {len(vertices)}")
    
    if ft.max() >= len(vt):
        raise ValueError(f"Texture face indices exceed texture coordinate count: max ft index {ft.max()}, vt count {len(vt)}")
    
    expanded_vertices = []
    expanded_uvs = []
    new_faces = []
    vertex_map = {}
    next_vertex_idx = 0
    
    for face_idx in range(len(faces)):
        face = faces[face_idx]
        tex_face = ft[face_idx]
        new_face = []
        
        for i in range(3):
            vertex_idx = face[i]
            uv_idx = tex_face[i]
            key = (vertex_idx, uv_idx)
            
            if key not in vertex_map:
                expanded_vertices.append(vertices[vertex_idx])
                expanded_uvs.append(vt[uv_idx])
                vertex_map[key] = next_vertex_idx
                next_vertex_idx += 1
            
            new_face.append(vertex_map[key])
        
        new_faces.append(new_face)
    
    vertices = np.array(expanded_vertices, dtype=np.float32)
    vt = np.array(expanded_uvs, dtype=np.float32)
    faces = np.array(new_faces, dtype=np.uint32)
    
    vertex_data = vertices.tobytes()
    texcoord_data = vt.tobytes()
    indices_data = faces.flatten().tobytes()
    
    vertex_size = len(vertex_data)
    texcoord_size = len(texcoord_data)
    indices_size = len(indices_data)
    
    def align_to_4(size):
        return (size + 3) & ~3
    
    vertex_aligned = align_to_4(vertex_size)
    texcoord_aligned = align_to_4(texcoord_size)
    
    buffer_data = bytearray()
    buffer_data.extend(vertex_data)
    buffer_data.extend(b'\x00' * (vertex_aligned - vertex_size))
    
    texcoord_offset = len(buffer_data)
    buffer_data.extend(texcoord_data)
    buffer_data.extend(b'\x00' * (texcoord_aligned - texcoord_size))
    
    indices_offset = len(buffer_data)
    buffer_data.extend(indices_data)
    
    buffer = Buffer(byteLength=len(buffer_data))
    buffer_index = len(scene.buffers)
    scene.buffers.append(buffer)
    
    vertex_buffer_view = BufferView(
        buffer=buffer_index,
        byteOffset=0,
        byteLength=vertex_size,
        target=ARRAY_BUFFER
    )
    vertex_buffer_view_index = len(scene.bufferViews)
    scene.bufferViews.append(vertex_buffer_view)
    
    texcoord_buffer_view = BufferView(
        buffer=buffer_index,
        byteOffset=texcoord_offset,
        byteLength=texcoord_size,
        target=ARRAY_BUFFER
    )
    texcoord_buffer_view_index = len(scene.bufferViews)
    scene.bufferViews.append(texcoord_buffer_view)
    
    indices_buffer_view = BufferView(
        buffer=buffer_index,
        byteOffset=indices_offset,
        byteLength=indices_size,
        target=ELEMENT_ARRAY_BUFFER
    )
    indices_buffer_view_index = len(scene.bufferViews)
    scene.bufferViews.append(indices_buffer_view)
    
    vertex_accessor = Accessor(
        bufferView=vertex_buffer_view_index,
        componentType=FLOAT,
        count=len(vertices),
        type=VEC3,
        min=vertices.min(axis=0).tolist(),
        max=vertices.max(axis=0).tolist()
    )
    vertex_accessor_index = len(scene.accessors)
    scene.accessors.append(vertex_accessor)
    
    texcoord_accessor = Accessor(
        bufferView=texcoord_buffer_view_index,
        componentType=FLOAT,
        count=len(vt),
        type=VEC2,
        min=vt.min(axis=0).tolist(),
        max=vt.max(axis=0).tolist()
    )
    texcoord_accessor_index = len(scene.accessors)
    scene.accessors.append(texcoord_accessor)
    
    indices_accessor = Accessor(
        bufferView=indices_buffer_view_index,
        componentType=UNSIGNED_INT,
        count=len(faces.flatten()),
        type=SCALAR
    )
    indices_accessor_index = len(scene.accessors)
    scene.accessors.append(indices_accessor)
    
    pil_image = Image.fromarray(texture_map, 'RGB')
    buffer_io = BytesIO()
    pil_image.save(buffer_io, format='PNG')
    image_data = buffer_io.getvalue()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"
    
    gltf_image = GLTFImage(uri=image_uri)
    image_index = len(scene.images)
    scene.images.append(gltf_image)
    
    sampler = Sampler()
    sampler_index = len(scene.samplers)
    scene.samplers.append(sampler)
    
    texture = Texture(source=image_index, sampler=sampler_index)
    texture_index = len(scene.textures)
    scene.textures.append(texture)
    
    pbr_metallic_roughness = PbrMetallicRoughness(
        baseColorTexture={"index": texture_index},
        metallicFactor=0.0,
        roughnessFactor=0.8
    )
    material = Material(
        name=material_name,
        pbrMetallicRoughness=pbr_metallic_roughness
    )
    material_index = len(scene.materials)
    scene.materials.append(material)
    
    primitive = Primitive(
        attributes=Attributes(
            POSITION=vertex_accessor_index,
            TEXCOORD_0=texcoord_accessor_index
        ),
        indices=indices_accessor_index,
        material=material_index
    )
    
    mesh = Mesh(name=mesh_name, primitives=[primitive])
    mesh_index = len(scene.meshes)
    scene.meshes.append(mesh)
    
    node = Node(mesh=mesh_index)
    node_index = len(scene.nodes)
    scene.nodes.append(node)
    
    scene.scenes[0].nodes.append(node_index)
    
    if not hasattr(scene, '_buffer_data'):
        scene._buffer_data = {}
    scene._buffer_data[buffer_index] = buffer_data
    
    return mesh_index

def save_glb_scene(save_path, scene=None):
    global _current_scene
    
    if scene is None:
        scene = _current_scene
    
    if scene is None:
        raise ValueError("No scene available. Call create_glb_scene() first.")
    
    if hasattr(scene, '_buffer_data') and scene._buffer_data:
        total_size = 0
        buffer_info = []
        
        for i, buffer_data in scene._buffer_data.items():
            if i < len(scene.buffers):
                aligned_size = (len(buffer_data) + 3) & ~3
                buffer_info.append((i, total_size, len(buffer_data), aligned_size, buffer_data))
                total_size += aligned_size
        
        unified_buffer = bytearray(total_size)
        
        for buffer_idx, offset, original_size, aligned_size, buffer_data in buffer_info:
            unified_buffer[offset:offset + original_size] = buffer_data
            if aligned_size > original_size:
                unified_buffer[offset + original_size:offset + aligned_size] = b'\x00' * (aligned_size - original_size)
            
            for bv in scene.bufferViews:
                if bv.buffer == buffer_idx:
                    bv.byteOffset += offset
                    bv.buffer = 0
        
        scene.buffers = [Buffer(byteLength=total_size)]
        scene.set_binary_blob(unified_buffer)
    
    scene.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="path to config file", type=str)
    parser.add_argument("--timestamp", required=True, help="timestamp or 'latest' to use most recent", type=str)
    args = parser.parse_args()

    conf = args.conf
    timestamp = args.timestamp
    conf = ConfigFactory.parse_file(conf)
    expname = conf.get_string('train.expname')

    exps_folder_name = "exps"
    expdir = os.path.join('./', exps_folder_name, expname)
    
    # Handle 'latest' timestamp by finding the most recent timestamp directory
    if timestamp == 'latest':
        if os.path.exists(expdir):
            timestamps = os.listdir(expdir)
            if len(timestamps) == 0:
                raise ValueError(f"No timestamp directories found in {expdir}")
            else:
                timestamp = sorted(timestamps)[-1]
                print(f"Using latest timestamp: {timestamp}")
        else:
            raise ValueError(f"Experiment directory {expdir} does not exist")
    
    plots_dir = os.path.join(expdir, timestamp, 'plots')

    mesh_paths = sorted(glob.glob(os.path.join(plots_dir, "surface_*.obj")))
    mesh_paths = sorted(mesh_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
    print("mesh_paths: ", mesh_paths)

    create_glb_scene()

    tex_dict_list = []
    
    for mesh_path in mesh_paths:
        mesh_p3d = load_objs_as_meshes([mesh_path])
        tex_dict = load_tex_dict_from_tex_mesh_p3d(mesh_p3d)
        tex_dict_list.append(tex_dict)
    
    all_meshes = [
        trimesh.Trimesh(
            tex_dict["vertices"], tex_dict["faces"], process=False
        ) for tex_dict in tex_dict_list
    ]
    translation_dict_path = os.path.abspath(os.path.join(plots_dir, 'translation_dict.pkl'))
    if os.path.exists(translation_dict_path):
        with open(translation_dict_path, 'rb') as f:
            translation_dict = pickle.load(f)
        for mesh_i in range(1, len(all_meshes)):
            all_meshes[mesh_i].vertices = all_meshes[mesh_i].vertices + translation_dict[mesh_i].reshape(1, 3)
    # all_meshes = solve_intersection(all_meshes)
    tex_dict_list = [
        {
            "vertices": all_meshes[tex_dict_i].vertices,
            "faces": all_meshes[tex_dict_i].faces,
            "vt": tex_dict_list[tex_dict_i]["vt"],
            "ft": tex_dict_list[tex_dict_i]["ft"],
            "texture_map": tex_dict_list[tex_dict_i]["texture_map"]
        }
        for tex_dict_i in range(len(tex_dict_list))
    ]
        
    for tex_dict in tex_dict_list:
        add_textured_mesh_to_glb_scene(tex_dict)
    
    save_glb_scene(os.path.join(plots_dir, "scene.glb"))
    print("saved to ", os.path.join(plots_dir, "scene.glb"))
