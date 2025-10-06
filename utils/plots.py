import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

from utils import rend_util
from utils.general import trans_topil, get_lcc_mesh

import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots

import os
import json
import psutil
from tqdm import tqdm
import gc
import pymeshlab as pml
def remesh(verts, faces):
    triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    edge_20 = triangles[:, 0] - triangles[:, 2]
    edge_len = np.sqrt(np.sum(edge_01 ** 2, axis=1))
    edge_len += np.sqrt(np.sum(edge_12 ** 2, axis=1))
    edge_len += np.sqrt(np.sum(edge_20 ** 2, axis=1))
    mean_edge_len = np.mean(edge_len / 3)

    pml_mesh = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(pml_mesh, 'mesh')

    ms.apply_filter('meshing_isotropic_explicit_remeshing', targetlen=pml.PureValue(mean_edge_len))

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return verts, faces

def get_face_normals(verts, faces):
    triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    normals = np.cross(edge_01, edge_12)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals

def get_vert_normals(verts, faces, face_normals):
    # use batch computation instead of for loop
    vert_normals = np.zeros_like(verts)
    for i in range(3):
        vert_normals[faces[:, i]] += face_normals
    vert_normals /= np.linalg.norm(vert_normals, axis=1)[:, None]
    return vert_normals

def simplify_mesh(verts, faces, target_faces):
    pml_mesh = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(pml_mesh, 'mesh')

    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_faces)

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return verts, faces

def plot(implicit_network, indices, plot_data, path, epoch, iter, img_res, plot_nimgs, resolution, grid_boundary, plot_mesh=True, obj_bbox_dict=None,  level=0):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], plot_data['normal_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot sem maps
        plot_sem_maps(plot_data['seg_map'], plot_data['seg_gt'], path, epoch, plot_nimgs, img_res, indices)

        # concat output images to single large image
        images = []
        for name in ["rendering", "depth", "normal", "sem"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))        

        images = np.concatenate(images, axis=1)
        cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)

    if plot_mesh:
        # plot each mesh
        sem_num = implicit_network.d_out
        _ = get_semantic_surface_trace(path=path,
                                        epoch=epoch,
                                        iter=iter,
                                        sdf = lambda x: implicit_network.get_sdf_raw(x),
                                        resolution=512,
                                        grid_boundary=grid_boundary,
                                        level=level,
                                        num = sem_num,
                                        obj_bbox_dict=obj_bbox_dict
                                        )
        # print("get_surface_trace")
        _ = get_surface_trace(path=path,
                            epoch=epoch,
                            sdf=lambda x: implicit_network.get_sdf_vals(x),
                            resolution=768,
                            grid_boundary=grid_boundary,
                            level=level
                            )

def plot_color_mesh(implicit_network, pr_network, indices, plot_data, path, epoch, iter, img_res, plot_nimgs, resolution,
                    grid_boundary, plot_mesh=True, obj_bbox_dict=None,  level=0, shift=False, trainer=None, color_mesh=False):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], plot_data['normal_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot sem maps
        plot_sem_maps(plot_data['seg_map'], plot_data['seg_gt'], path, epoch, plot_nimgs, img_res, indices)

        # concat output images to single large image
        images = []
        for name in ["rendering", "depth", "normal", "sem"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))

        images = np.concatenate(images, axis=1)
        cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)

    if plot_mesh:
        # plot each mesh
        gc.collect()
        del plot_data
        sem_num = implicit_network.d_out
        if shift:
            sdf_func = lambda x: implicit_network.get_shift_sdf_raw(x)
        else:
            sdf_func = lambda x: implicit_network.get_sdf_raw(x)

        _ = get_semantic_surface_trace_colors_mask_filter(path=path,
                                        epoch=epoch,
                                        iter=iter,
                                        sdf = sdf_func,
                                       color = lambda x, y, z: pr_network.get_colors_from_point_rays_obj_offset(x, y, z),
                                        resolution=512,
                                        grid_boundary=grid_boundary,
                                        level=level,
                                        num = sem_num,
                                        obj_bbox_dict=obj_bbox_dict,
                                        trainer=trainer,
                                        color_mesh=color_mesh
                                        )
        # print("get_surface_trace")
        _ = get_surface_trace(path=path,
                            epoch=epoch,
                            sdf=lambda x: implicit_network.get_sdf_vals(x),
                            resolution=768,
                            grid_boundary=grid_boundary,
                            level=level
                            )

def get_semantic_surface_trace_colors_obj(path, iter, sdf, color, idx, resolution=100, grid_boundary=[-2.0, 2.0],
                               level=0, obj_bbox_dict=None):
    if obj_bbox_dict is not None and idx != 0:  # 0 is bg
        obj_bbox = obj_bbox_dict[idx]
        grid = get_grid_bbox(resolution, obj_bbox)
    else:
        grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    with torch.no_grad():
        z_all = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)

    z = z_all[:, idx]
    if (not (np.min(z) > level or np.max(z) < level)):
        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                             grid['xyz'][2].shape[0]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][1][2] - grid['xyz'][1][1],
                     grid['xyz'][2][2] - grid['xyz'][2][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        if idx == 0:
            if faces.shape[0] > 200000:
                verts, faces = simplify_mesh(verts, faces, 200000)
        else:
            if faces.shape[0] > 20000:
                verts, faces = simplify_mesh(verts, faces, 20000)
        verts, faces = remesh(verts, faces)
        vert_normals = get_vert_normals(verts, faces, get_face_normals(verts, faces))

        mean_face_dist = float(np.mean(np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=-1)))
        color_all = []

        for i, (vert_pnts, vert_normal) in enumerate(
                zip(torch.split(torch.from_numpy(verts.reshape(-1, 3).copy()), 1024, dim=0),
                    torch.split(torch.from_numpy(vert_normals.reshape(-1, 3).copy()), 1024, dim=0))):
            ray_start = vert_pnts + vert_normal * mean_face_dist * 1
            ray_dir = -vert_normal
            color_all.append(color(
                ray_start.cuda().float().detach(),
                ray_dir.cuda().float().detach()).detach().cpu().numpy())
            torch.cuda.empty_cache()
        verts_color = np.concatenate(color_all, axis=0)
        verts_color = np.clip(verts_color * 255., 0, 255)

        meshexport = trimesh.Trimesh(verts, faces, vertex_colors=verts_color, process=False)
        meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, iter, idx), 'ply')

def generate_specific_obj_mesh(implicit_network, pr_network, save_path, iter, obj_idx, resolution, grid_boundary, level=0, obj_bbox_dict=None):
    get_semantic_surface_trace_colors_obj(
        path=save_path,
        iter=iter,
        sdf=lambda x: implicit_network.get_shift_sdf_raw(x),
        color=lambda x, y: pr_network.get_colors_from_point_rays(x, y),
        idx=obj_idx,
        resolution=resolution,
        grid_boundary=grid_boundary,
        level=level,
        obj_bbox_dict=obj_bbox_dict
    )

def plot_(implicit_network, indices, plot_data, path, epoch, iter, img_res, plot_nimgs, resolution, grid_boundary, plot_mesh=True, obj_bbox_dict=None,  level=0):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], plot_data['normal_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot sem maps
        plot_sem_maps(plot_data['seg_map'], plot_data['seg_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth_un maps
        plot_uncertainty_maps(plot_data['depth_un_map'], path, epoch, img_res, indices, "depth")

        # plot normal_un maps
        plot_uncertainty_maps(plot_data['normal_un_map'], path, epoch, img_res, indices, "normal")

        # plot physical un maps
        plot_uncertainty_maps(plot_data['phy_un_map'], path, epoch, img_res, indices, "physical")

        # concat output images to single large image
        images = []
        for name in ["rendering", "depth", "normal", "sem"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))

        images = np.concatenate(images, axis=1)
        cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)

        # cat uncertainty map
        images_uncertainty = []
        for name in ["depth_uncertainty", "normal_uncertainty", "physical_uncertainty"]:
            images_uncertainty.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))

        images_uncertainty = np.concatenate(images_uncertainty, axis=1)
        cv2.imwrite('{0}/merge_uncertainty_{1}_{2}.png'.format(path, epoch, indices[0]), images_uncertainty)

    if plot_mesh:
        # plot each mesh
        sem_num = implicit_network.d_out
        _ = get_semantic_surface_trace_(path=path,
                                        epoch=epoch,
                                        iter=iter,
                                        sdf = lambda x: implicit_network.get_outputs_and_indices(x),
                                        resolution=512,
                                        grid_boundary=grid_boundary,
                                        level=level,
                                        num = sem_num,
                                        obj_bbox_dict=obj_bbox_dict
                                        )
        _ = get_surface_trace(path=path,
                            epoch=epoch,
                            sdf=lambda x: implicit_network.get_sdf_vals(x),
                            resolution=768,
                            grid_boundary=grid_boundary,
                            level=level
                            )

avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')


@torch.no_grad()
def get_surface_sliding(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    assert resolution % 256 == 0
    resN = resolution
    cropN = 256
    level = 0
    N = resN // cropN

    grid_min = [grid_boundary[0], grid_boundary[0], grid_boundary[0]]
    grid_max = [grid_boundary[1], grid_boundary[1], grid_boundary[1]]
    xs = np.linspace(grid_min[0], grid_max[0], N+1)
    ys = np.linspace(grid_min[1], grid_max[1], N+1)
    zs = np.linspace(grid_min[2], grid_max[2], N+1)

    print(xs)
    print(ys)
    print(zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i+1]
                y_min, y_max = ys[j], ys[j+1]
                z_min, z_max = zs[k], zs[k+1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
                
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z
            
                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):            
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]
                
                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min)/cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()
                    
                    if mask is None:    
                        pts_sdf = evaluate(pts)
                    else:                    
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        #import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.

                z = pts_sdf.detach().cpu().numpy()

                if (not (np.min(z) > level or np.max(z) < level)):
                    z = z.astype(np.float32)
                    verts, faces, normals, values = measure.marching_cubes(
                    volume=z.reshape(cropN, cropN, cropN), #.transpose([1, 0, 2]),
                    level=level,
                    spacing=(
                            (x_max - x_min)/(cropN-1),
                            (y_max - y_min)/(cropN-1),
                            (z_max - z_min)/(cropN-1) ))
                    print(np.array([x_min, y_min, z_min]))
                    print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    print(verts.min(), verts.max())
                    
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    #meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)

    if return_mesh:
        return combined
    else:
        combined.export('{0}/surface_{1}_whole_eval.ply'.format(path, epoch), 'ply') 
        print(f'surface_{epoch}_whole_eval.ply save to {path}')


def get_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    # print("0 get_surface_trace")
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    # print("1 get_surface_trace")
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                             grid['xyz'][2].shape[0]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        # print("2 get_surface_trace")
        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        # print("3 get_surface_trace")
        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}_whole.ply'.format(path, epoch), 'ply')
        print(f'surface_{epoch}_whole.ply save to {path}')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_semantic_surface_trace(path, epoch, iter, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0, num=0, obj_bbox_dict=None):

    for idx in range(num):

        if obj_bbox_dict is not None and idx != 0:          # 0 is bg
            obj_bbox = obj_bbox_dict[idx]
            grid = get_grid_bbox(resolution, obj_bbox)
        else:
            grid = get_grid_uniform(resolution, grid_boundary)
        points = grid['grid_points']

        with torch.no_grad():
            z_all = []
            for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
                z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
            z_all = np.concatenate(z_all, axis=0)

        z = z_all[:, idx]
        if (not (np.min(z) > level or np.max(z) < level)):

            z = z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                                 grid['xyz'][2].shape[0]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][1][2] - grid['xyz'][1][1],
                         grid['xyz'][2][2] - grid['xyz'][2][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            print("processing mesh {0} ...".format(idx))
            if idx == 0:
                if faces.shape[0] > 1000000:
                    verts, faces = simplify_mesh(verts, faces, 1000000)
            else:
                if faces.shape[0] > 100000:
                    verts, faces = simplify_mesh(verts, faces, 100000)
            verts, faces = remesh(verts, faces)
            normals = get_vert_normals(verts, faces, get_face_normals(verts, faces))

            I, J, K = faces.transpose()

            traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=I, j=J, k=K, name='implicit_surface',
                                color='#ffffff', opacity=1.0, flatshading=False,
                                lighting=dict(diffuse=1, ambient=0, specular=0),
                                lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

            meshexport = trimesh.Trimesh(verts, faces)
            meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')
            print(f'surface_{epoch}_{idx}.ply save to {path}')

        # export mesh bbox
        if iter > 60000 and iter < 150000:        # save bbox after 60000 iterations, 150000 begin physical simulation
            
            z_floor_txt_path = os.path.join(path, 'z_floor.txt')
            if os.path.exists(z_floor_txt_path):
                print('not change object bbox after refine object bbox using z_floor')
            else:
                bbox_root_path = os.path.join(path, 'bbox')
                os.makedirs(bbox_root_path, exist_ok=True)
                bbox_json_path = os.path.join(bbox_root_path, f'bbox_{idx}.json')
                if os.path.exists(bbox_json_path):
                    os.remove(bbox_json_path)
                x_min, x_max = meshexport.vertices[:, 0].min() - 0.1, meshexport.vertices[:, 0].max() + 0.1
                y_min, y_max = meshexport.vertices[:, 1].min() - 0.1, meshexport.vertices[:, 1].max() + 0.1
                z_min, z_max = meshexport.vertices[:, 2].min() - 0.1, meshexport.vertices[:, 2].max() + 0.1
                x_min, x_max = max(x_min, -1.0), min(x_max, 1.0)
                y_min, y_max = max(y_min, -1.0), min(y_max, 1.0)
                z_min, z_max = max(z_min, -1.0), min(z_max, 1.0)
                obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                with open(bbox_json_path, 'w') as f:
                    json.dump(obj_bbox, f)
                print(f'bbox_{idx}.json save to {bbox_root_path}')
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

        torch.cuda.empty_cache()
        # report GPU memory allocation
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB

        print(f"allocated {allocated_mem:.2f}GB / total {total_mem:.2f}GB")
    # print("finish get_semantic_surface_trace")

    if return_mesh:
        return meshexport
    return traces

# def get_vertex_normals(verts, faces, face_normals):
#     # Initialize vertex normals with zeros
#     print("verts: ", verts.shape)
#     print("faces: ", faces.shape)
#     print("face_normals: ", face_normals.shape)
#     vert_normals = np.zeros_like(verts)
#
#     np.add.at(vert_normals, faces, face_normals[:, np.newaxis, :])
#
#     # Normalize the accumulated normals
#     norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
#     norms[norms == 0] = 1  # Avoid division by zero
#     vert_normals /= norms
#
#     return vert_normals

def get_semantic_surface_trace_colors(path, epoch, iter, sdf, color, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False,
                               level=0, num=0, obj_bbox_dict=None):
    for idx in range(num):

        if obj_bbox_dict is not None and idx != 0:  # 0 is bg
            obj_bbox = obj_bbox_dict[idx]
            grid = get_grid_bbox(resolution, obj_bbox)
        else:
            grid = get_grid_uniform(resolution, grid_boundary)
        points = grid['grid_points']

        with torch.no_grad():
            z_all = []
            for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
                z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
            z_all = np.concatenate(z_all, axis=0)

        z = z_all[:, idx]
        if (not (np.min(z) > level or np.max(z) < level)):
            z = z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                                 grid['xyz'][2].shape[0]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][1][2] - grid['xyz'][1][1],
                         grid['xyz'][2][2] - grid['xyz'][2][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            # verts, faces, _, _ = get_lcc_mesh(verts, faces)

            if idx == 0:
                if faces.shape[0] > 200000:
                    verts, faces = simplify_mesh(verts, faces, 200000)
            else:
                if faces.shape[0] > 20000:
                    verts, faces = simplify_mesh(verts, faces, 20000)
            verts, faces = remesh(verts, faces)
            vert_normals = get_vert_normals(verts, faces, get_face_normals(verts, faces))

            # vert_normals = get_vertex_normals(verts, faces, normals)
            # vert_normals = -normals
            # calculate the average distance between vertices pairs
            mean_face_dist = float(np.mean(np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=-1)))
            color_all = []
            # for i, vert_pnts, vert_normal in zip(range(verts.shape[0]), verts, vert_normals):
            for i , (vert_pnts, vert_normal) in enumerate(zip(torch.split(torch.from_numpy(verts.reshape(-1, 3).copy()), 1024, dim=0), torch.split(torch.from_numpy(vert_normals.reshape(-1, 3).copy()), 1024, dim=0))):
                ray_start = vert_pnts + vert_normal * mean_face_dist * 1
                ray_dir = -vert_normal
                color_all.append(color(
                    ray_start.cuda().float().detach(),
                    ray_dir.cuda().float().detach(),
                    idx).detach().cpu().numpy())
                torch.cuda.empty_cache()
            verts_color = np.concatenate(color_all, axis=0)
            verts_color = np.clip(verts_color*255., 0, 255)


            I, J, K = faces.transpose()

            traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=I, j=J, k=K, name='implicit_surface',
                                color='#ffffff', opacity=1.0, flatshading=False,
                                lighting=dict(diffuse=1, ambient=0, specular=0),
                                lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

            # meshexport = trimesh.Trimesh(verts, faces, vertex_normals=-normals, vertex_colors=verts_color, process=False)
            meshexport = trimesh.Trimesh(verts, faces, vertex_colors=verts_color, process=False)
            meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')
            print(f'surface_color_{epoch}_{idx}.ply save to {path}')

        # export mesh bbox
        if iter > 60000 and iter < 150000:  # save bbox after 60000 iterations, 150000 begin physical simulation

            z_floor_txt_path = os.path.join(path, 'z_floor.txt')
            if os.path.exists(z_floor_txt_path):
                print('not change object bbox after refine object bbox using z_floor')
            else:
                bbox_root_path = os.path.join(path, 'bbox')
                os.makedirs(bbox_root_path, exist_ok=True)
                bbox_json_path = os.path.join(bbox_root_path, f'bbox_{idx}.json')
                if os.path.exists(bbox_json_path):
                    os.remove(bbox_json_path)
                x_min, x_max = meshexport.vertices[:, 0].min() - 0.1, meshexport.vertices[:, 0].max() + 0.1
                y_min, y_max = meshexport.vertices[:, 1].min() - 0.1, meshexport.vertices[:, 1].max() + 0.1
                z_min, z_max = meshexport.vertices[:, 2].min() - 0.1, meshexport.vertices[:, 2].max() + 0.1
                x_min, x_max = max(x_min, -1.0), min(x_max, 1.0)
                y_min, y_max = max(y_min, -1.0), min(y_max, 1.0)
                z_min, z_max = max(z_min, -1.0), min(z_max, 1.0)
                obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                with open(bbox_json_path, 'w') as f:
                    json.dump(obj_bbox, f)
                print(f'bbox_{idx}.json save to {bbox_root_path}')
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)

        torch.cuda.empty_cache()
        # report GPU memory allocation
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB

        print(f"allocated {allocated_mem:.2f}GB / total {total_mem:.2f}GB")
    # print("finish get_semantic_surface_trace")

    if return_mesh:
        return meshexport
    return traces


def create_3d_partitions(grid_boundary, resolution, partition_size=128):
    """
    Create 3D partitions for space subdivision when resolution is high.
    
    Args:
        grid_boundary: [min, max] boundary for the grid
        resolution: total resolution for the grid
        partition_size: maximum resolution per partition (default 128)
    
    Returns:
        List of partition boundaries and their resolutions
    """
    if resolution <= 256:
        # No partitioning needed
        return [{'boundary': grid_boundary, 'resolution': resolution}]
    
    # Calculate number of partitions needed per axis
    partitions_per_axis = int(np.ceil(resolution / partition_size))
    actual_partition_res = int(np.ceil(resolution / partitions_per_axis))
    
    # Create partition boundaries
    grid_min, grid_max = grid_boundary[0], grid_boundary[1]
    grid_range = grid_max - grid_min
    partition_boundaries = []
    
    for i in range(partitions_per_axis):
        for j in range(partitions_per_axis):
            for k in range(partitions_per_axis):
                # Calculate boundaries for this partition
                x_min = grid_min + (i * grid_range) / partitions_per_axis
                x_max = grid_min + ((i + 1) * grid_range) / partitions_per_axis
                y_min = grid_min + (j * grid_range) / partitions_per_axis
                y_max = grid_min + ((j + 1) * grid_range) / partitions_per_axis
                z_min = grid_min + (k * grid_range) / partitions_per_axis
                z_max = grid_min + ((k + 1) * grid_range) / partitions_per_axis
                
                partition_boundaries.append({
                    'boundary': [[x_min, y_min, z_min], [x_max, y_max, z_max]],
                    'resolution': actual_partition_res,
                    'partition_id': (i, j, k)
                })
    
    return partition_boundaries


def create_3d_partitions_bbox(obj_bbox, resolution, partition_size=128):
    """
    Create 3D partitions for bounding box subdivision when resolution is high.
    
    Args:
        obj_bbox: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        resolution: total resolution for the grid
        partition_size: maximum resolution per partition (default 128)
    
    Returns:
        List of partition bounding boxes and their resolutions
    """
    if resolution <= 256:
        # No partitioning needed
        return [{'bbox': obj_bbox, 'resolution': resolution}]
    
    # Calculate number of partitions needed per axis
    partitions_per_axis = int(np.ceil(resolution / partition_size))
    actual_partition_res = int(np.ceil(resolution / partitions_per_axis))
    
    # Extract bbox boundaries
    min_bounds = np.array(obj_bbox[0])
    max_bounds = np.array(obj_bbox[1])
    ranges = max_bounds - min_bounds
    
    partition_bboxes = []
    
    for i in range(partitions_per_axis):
        for j in range(partitions_per_axis):
            for k in range(partitions_per_axis):
                # Calculate boundaries for this partition
                partition_min = min_bounds + np.array([i, j, k]) * ranges / partitions_per_axis
                partition_max = min_bounds + np.array([i + 1, j + 1, k + 1]) * ranges / partitions_per_axis
                
                partition_bboxes.append({
                    'bbox': [partition_min.tolist(), partition_max.tolist()],
                    'resolution': actual_partition_res,
                    'partition_id': (i, j, k)
                })
    
    return partition_bboxes


def test_partitioning():
    """
    Test function to verify 3D partitioning works correctly.
    """
    # Test uniform grid partitioning
    partitions = create_3d_partitions([-1.0, 1.0], 300)
    print(f"Created {len(partitions)} partitions for resolution 300")
    
    # Test bbox partitioning  
    bbox_partitions = create_3d_partitions_bbox([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], 400)
    print(f"Created {len(bbox_partitions)} bbox partitions for resolution 400")
    
    # Test no partitioning case
    no_partitions = create_3d_partitions([-1.0, 1.0], 128)
    print(f"Created {len(no_partitions)} partitions for resolution 128 (should be 1)")


def get_semantic_surface_trace_colors_mask_filter(path, epoch, iter, sdf, color, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False,
                               level=0, num=0, obj_bbox_dict=None, trainer=None, color_mesh=False):
    """
    Get semantic surface trace with colors and mask filtering.
    
    When resolution > 256, this function automatically uses 3D space partitioning to:
    1. Split the 3D space into smaller partitions (max 128^3 resolution each)
    2. Apply marching cubes to each partition independently 
    3. Merge the resulting meshes using trimesh.grouping.merge_vertices
    
    This approach reduces memory usage and prevents out-of-memory errors for high resolutions.
    
    Args:
        path: Output path for mesh files
        epoch: Current epoch number  
        iter: Current iteration number
        sdf: SDF function
        color: Color function
        resolution: Grid resolution (partitioned if > 256)
        grid_boundary: Grid boundary for uniform grids
        return_mesh: Whether to return mesh object
        level: Isosurface level for marching cubes
        num: Number of objects to process
        obj_bbox_dict: Dictionary of object bounding boxes
        trainer: Trainer object with mask_filter method
        color_mesh: Whether to apply coloring to mesh
        
    Returns:
        Mesh object if return_mesh=True, otherwise traces
    """
    
    for idx in tqdm(range(num)):
        current_resolution = resolution if idx == 0 else resolution // 2
        
        # Determine if we need partitioning
        if current_resolution > 256:
            # Use partitioning approach
            if obj_bbox_dict is not None and idx != 0:  # 0 is bg
                obj_bbox = obj_bbox_dict[idx]
                partitions = create_3d_partitions_bbox(obj_bbox, current_resolution)
            else:
                partitions = create_3d_partitions(grid_boundary, current_resolution)
            
            # Process each partition and collect meshes
            partition_meshes = []
            
            for partition in partitions:
                if obj_bbox_dict is not None and idx != 0:
                    grid = get_grid_bbox(partition['resolution'], partition['bbox'])
                else:
                    # Convert partition boundary format for get_grid_uniform
                    part_min = partition['boundary'][0]
                    part_max = partition['boundary'][1]
                    # get_grid_uniform expects [min, max] format but creates cubic grids
                    # We need to create a custom grid for the partition
                    part_boundary = [part_min[0], part_max[0]]  # Using x-axis range for uniform spacing
                    grid = get_grid_uniform(partition['resolution'], part_boundary)
                    
                    # Override the xyz coordinates to match the actual partition boundaries
                    x = np.linspace(part_min[0], part_max[0], partition['resolution'])
                    y = np.linspace(part_min[1], part_max[1], partition['resolution'])
                    z = np.linspace(part_min[2], part_max[2], partition['resolution'])
                    
                    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
                    
                    grid = {"grid_points": grid_points,
                            "shortest_axis_length": max(part_max[0] - part_min[0], part_max[1] - part_min[1], part_max[2] - part_min[2]),
                            "xyz": [x, y, z],
                            "shortest_axis_index": 0}
                
                points = grid['grid_points']

                with torch.no_grad():
                    z_all = []
                    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
                    z_all = np.concatenate(z_all, axis=0)

                z = z_all[:, idx]
                if (not (np.min(z) > level or np.max(z) < level)):
                    z = z.astype(np.float32)

                    try:
                        verts, faces, normals, values = measure.marching_cubes(
                            volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                                             grid['xyz'][2].shape[0]),
                            level=level,
                            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                     grid['xyz'][1][2] - grid['xyz'][1][1],
                                     grid['xyz'][2][2] - grid['xyz'][2][1]))

                        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
                        
                        # Create mesh for this partition
                        if len(faces) > 0:
                            partition_mesh = trimesh.Trimesh(verts, faces, process=False)
                            partition_meshes.append(partition_mesh)
                    except Exception as e:
                        print(f"Warning: Marching cubes failed for partition {partition.get('partition_id', '')}: {e}")
                        continue
            
            # Merge all partition meshes
            if partition_meshes:
                if len(partition_meshes) == 1:
                    combined_mesh = partition_meshes[0]
                else:
                    # Combine all meshes
                    combined_mesh = trimesh.util.concatenate(partition_meshes)
                    # Merge vertices to remove duplicates at partition boundaries
                    combined_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
                
                verts = combined_mesh.vertices
                faces = combined_mesh.faces
            else:
                # No valid meshes found
                continue
                
        else:
            # Original approach for low resolution
            if obj_bbox_dict is not None and idx != 0:  # 0 is bg
                obj_bbox = obj_bbox_dict[idx]
                grid = get_grid_bbox(current_resolution, obj_bbox)
            else:
                grid = get_grid_uniform(current_resolution, grid_boundary)
            points = grid['grid_points']

            with torch.no_grad():
                z_all = []
                for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
                    z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
                z_all = np.concatenate(z_all, axis=0)

            z = z_all[:, idx]
            if (not (np.min(z) > level or np.max(z) < level)):
                z = z.astype(np.float32)

                verts, faces, normals, values = measure.marching_cubes(
                    volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                                     grid['xyz'][2].shape[0]),
                    level=level,
                    spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                             grid['xyz'][1][2] - grid['xyz'][1][1],
                             grid['xyz'][2][2] - grid['xyz'][2][1]))

                verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
            else:
                # No valid surface found
                continue

        # debug
        # meshexport = trimesh.Trimesh(verts, faces, process=False)
        # meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')

        verts, faces, _, _ = trainer.mask_filter(verts, faces, obj_i=idx)

        if idx == 0:
            if faces.shape[0] > 200000:
                verts, faces = simplify_mesh(verts, faces, 200000)
        else:
            if faces.shape[0] > 20000:
                verts, faces = simplify_mesh(verts, faces, 20000)
        verts, faces = remesh(verts, faces)

        face_normals = get_face_normals(verts, faces)
        mean_face_dist = float(np.mean(np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=-1)))
        face_centroids = np.mean(verts[faces.reshape(-1)].reshape(-1, 3, 3), axis=1).reshape(-1, 3)
        if color_mesh:
            color_all = []
            for i, (vert_pnts, vert_normal) in enumerate(
                    zip(torch.split(torch.from_numpy(face_centroids.reshape(-1, 3).copy()), 1024, dim=0),
                        torch.split(torch.from_numpy(face_normals.reshape(-1, 3).copy()), 1024, dim=0))):
                ray_start = vert_pnts + vert_normal * mean_face_dist * 0.2
                ray_dir = -vert_normal
                color_all.append(color(
                    ray_start.cuda().float().detach(),
                    ray_dir.cuda().float().detach(),
                    idx).detach().cpu().numpy())
                torch.cuda.empty_cache()
            face_color = np.concatenate(color_all, axis=0)
            face_color = np.clip(face_color, 0.0, 1.0)
            meshexport = trimesh.Trimesh(verts, faces, face_colors=face_color, process=False)
        else:
            meshexport = trimesh.Trimesh(verts, faces, face_colors=np.ones((faces.shape[0], 3)) * 0.5, process=False)

        meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')
        # print(f'surface_color_{epoch}_{idx}.ply save to {path}')

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        torch.cuda.empty_cache()
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB

    if return_mesh:
        return meshexport
    return traces


def get_semantic_surface_trace_(path, epoch, iter, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False,
                               level=0, num=0, obj_bbox_dict=None):
    for idx in range(num):

        if obj_bbox_dict is not None and idx != 0:  # 0 is bg
            obj_bbox = obj_bbox_dict[idx]
            grid = get_grid_bbox(resolution, obj_bbox)
        else:
            grid = get_grid_uniform(resolution, grid_boundary)
        points = grid['grid_points']

        z_all = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z_all.append(sdf(pnts.cuda())[4].detach().cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)

        z = z_all[:, idx]
        if (not (np.min(z) > level or np.max(z) < level)):
            z = z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                                 grid['xyz'][2].shape[0]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][1][2] - grid['xyz'][1][1],
                         grid['xyz'][2][2] - grid['xyz'][2][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if idx == 0:
                if faces.shape[0] > 200000:
                    verts, faces = simplify_mesh(verts, faces, 200000)
            else:
                if faces.shape[0] > 20000:
                    verts, faces = simplify_mesh(verts, faces, 20000)
            verts, faces = remesh(verts, faces)
            normals = get_vert_normals(verts, faces, get_face_normals(verts, faces))

            I, J, K = faces.transpose()

            traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=I, j=J, k=K, name='implicit_surface',
                                color='#ffffff', opacity=1.0, flatshading=False,
                                lighting=dict(diffuse=1, ambient=0, specular=0),
                                lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

            meshexport = trimesh.Trimesh(verts, faces)
            meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')
            print(f'surface_{epoch}_{idx}.ply save to {path}')

        # export mesh bbox
        if iter > 60000 and iter < 150000:  # save bbox after 60000 iterations, 150000 begin physical simulation

            z_floor_txt_path = os.path.join(path, 'z_floor.txt')
            if os.path.exists(z_floor_txt_path):
                print('not change object bbox after refine object bbox using z_floor')
            else:
                bbox_root_path = os.path.join(path, 'bbox')
                os.makedirs(bbox_root_path, exist_ok=True)
                bbox_json_path = os.path.join(bbox_root_path, f'bbox_{idx}.json')
                if os.path.exists(bbox_json_path):
                    os.remove(bbox_json_path)
                x_min, x_max = meshexport.vertices[:, 0].min() - 0.1, meshexport.vertices[:, 0].max() + 0.1
                y_min, y_max = meshexport.vertices[:, 1].min() - 0.1, meshexport.vertices[:, 1].max() + 0.1
                z_min, z_max = meshexport.vertices[:, 2].min() - 0.1, meshexport.vertices[:, 2].max() + 0.1
                x_min, x_max = max(x_min, -1.0), min(x_max, 1.0)
                y_min, y_max = max(y_min, -1.0), min(y_max, 1.0)
                z_min, z_max = max(z_min, -1.0), min(z_max, 1.0)
                obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                with open(bbox_json_path, 'w') as f:
                    json.dump(obj_bbox, f)
                print(f'bbox_{idx}.json save to {bbox_root_path}')

    if return_mesh:
        return meshexport
    return traces



def get_object_surface_trace(path, epoch, iter, sdf, resolution=100, return_mesh=False, level=0, obj_idx=0, obj_bbox=None):

    grid = get_grid_bbox(resolution, obj_bbox)
    points = grid['grid_points']

    z_all = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z_all = np.concatenate(z_all, axis=0)

    z = z_all[:, obj_idx]
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                            grid['xyz'][2].shape[0]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                    grid['xyz'][1][2] - grid['xyz'][1][1],
                    grid['xyz'][2][2] - grid['xyz'][2][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        if obj_idx == 0:
            if faces.shape[0] > 200000:
                verts, faces = simplify_mesh(verts, faces, 200000)
        else:
            if faces.shape[0] > 20000:
                verts, faces = simplify_mesh(verts, faces, 20000)
        verts, faces = remesh(verts, faces)
        normals = get_vert_normals(verts, faces, get_face_normals(verts, faces))

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces)
        meshexport.export('{0}/surface_{1}_{2}_stable.ply'.format(path, epoch, obj_idx), 'ply')
        print(f'surface_{epoch}_{obj_idx}_stable.ply save to {path}')

    if return_mesh:
        return meshexport
    return traces
        
def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace

def get_surface_high_res_mesh(sdf, resolution=100, grid_boundary=[-2.0, 2.0], level=0, take_components=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100, grid_boundary)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                         grid['xyz'][2].shape[0]),
        level=level,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    if take_components:
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][0].shape[0], grid_aligned['xyz'][1].shape[0],
                             grid_aligned['xyz'][2].shape[0]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_surface_by_grid(grid_params, sdf, resolution=100, level=0, higher_res=False):
    grid_params = grid_params * [[1.5], [1.0]]

    # params = PLOT_DICT[scan_id]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()

    if higher_res:
        # get low res mesh to sample point cloud
        grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)
        z = []
        points = grid['grid_points']

        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][0].shape[0], grid['xyz'][1].shape[0],
                             grid['xyz'][2].shape[0]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()

        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()

        grid_aligned = get_grid(helper.cpu(), resolution, eps=0.01)
    else:
        grid_aligned = get_grid(None, resolution, input_min=input_min, input_max=input_max, eps=0.0)

    grid_points = grid_aligned['grid_points']

    if higher_res:
        g = []
        for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][0].shape[0], grid_aligned['xyz'][1].shape[0],
                             grid_aligned['xyz'][2].shape[0]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        if higher_res:
            verts = torch.from_numpy(verts).cuda().float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()
        else:
            verts = verts + np.array([grid_aligned['xyz'][0][0], grid_aligned['xyz'][1][0], grid_aligned['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, normals)

        # CUTTING MESH ACCORDING TO THE BOUNDING BOX
        if higher_res:
            bb = grid_params
            transformation = np.eye(4)
            transformation[:3, 3] = (bb[1,:] + bb[0,:])/2.
            bounding_box = trimesh.creation.box(extents=bb[1,:] - bb[0,:], transform=transformation)

            meshexport = meshexport.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)

    return meshexport

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid_bbox(resolution, obj_bbox=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]):
    grid_min = [obj_bbox[0][0], obj_bbox[0][1], obj_bbox[0][2]]
    grid_max = [obj_bbox[1][0], obj_bbox[1][1], obj_bbox[1][2]]

    x = np.linspace(grid_min[0], grid_max[0], resolution)
    y = np.linspace(grid_min[1], grid_max[1], resolution)
    z = np.linspace(grid_min[2], grid_max[2], resolution)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "xyz": [x, y, z]}

def get_grid(points, resolution, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_normal_maps(normal_maps, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()
    normal_maps = torch.cat((normal_maps, ground_true), dim=0)
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/normal_{1}_{2}.png'.format(path, epoch, indices[0]))

    #import pdb; pdb.set_trace()
    #trans_topil(normal_maps_plot[0, :, :, 260:260+680]).save('{0}/2normal_{1}.png'.format(path, epoch))


def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, indices, exposure=False):
    ground_true = ground_true.cuda()

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if exposure:
        img.save('{0}/exposure_{1}_{2}.png'.format(path, epoch, indices[0]))
    else:
        img.save('{0}/rendering_{1}_{2}.png'.format(path, epoch, indices[0]))


def colored_data(x, cmap='jet', d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:,:,:3]).astype(np.uint8) # H, W, C

def plot_sem_maps(sem_maps, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()
    # import pdb; pdb.set_trace()
    sem_maps = torch.cat((sem_maps[..., None], ground_true), dim=0)
    sem_maps_plot = lin2img(sem_maps, img_res)

    tensor = torchvision.utils.make_grid(sem_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)[:,:,0]
    # import pdb; pdb.set_trace()
    tensor = colored_data(tensor)

    img = Image.fromarray(tensor)
    img.save('{0}/sem_{1}_{2}.png'.format(path, epoch, indices[0]))

    

def plot_depth_maps(depth_maps, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()
    depth_maps = torch.cat((depth_maps[..., None], ground_true), dim=0)
    depth_maps_plot = lin2img(depth_maps, img_res)
    depth_maps_plot = depth_maps_plot.expand(-1, 3, -1, -1)

    tensor = torchvision.utils.make_grid(depth_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    
    save_path = '{0}/depth_{1}_{2}.png'.format(path, epoch, indices[0])
    
    plt.imsave(save_path, tensor[:, :, 0], cmap='viridis')
    

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

def plot_uncertainty_maps(uncertainty_map, path, epoch, img_res, indices, type_str):
    uncertainty_map = uncertainty_map.cuda()
    
    uncertainty_map = uncertainty_map.reshape(img_res[0], img_res[1])
    uncertainty_map = uncertainty_map.cpu().detach().numpy()
    np.save('{0}/{1}_uncertainty_{2}_{3}.npy'.format(path, type_str, epoch, indices[0]), uncertainty_map)
    uncertainty_map = uncertainty_map / np.max(uncertainty_map)
    norm = Normalize(vmin=np.min(uncertainty_map), vmax=np.max(uncertainty_map))
    uncertainty_heatmap = plt.cm.hot(norm(uncertainty_map)).reshape(img_res[0], img_res[1], -1)
    uncertainty_heatmap = uncertainty_heatmap[:, :, :3]
    uncertainty_map_img = Image.fromarray((uncertainty_heatmap * 255).astype(np.uint8)).resize(img_res)
    uncertainty_map_img.save('{0}/{1}_uncertainty_{2}_{3}.png'.format(path, type_str, epoch, indices[0]))
