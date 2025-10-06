import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
import numpy as np
import json
import trimesh
import nvdiffrast.torch as dr
from PIL import Image
from skimage import measure

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.projections import get_projection_class
import rembg
from scipy import ndimage
from sklearn.cluster import DBSCAN, HDBSCAN

from collections import defaultdict, deque
import graphviz
import torchvision
import open3d
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from MVMeshRecon.remeshing.util.render import Renderer
from MVMeshRecon.remeshing.optimize import do_optimize
from MVMeshRecon.MeshRecon.optimize import simple_clean_mesh, to_pyml_mesh, geo_aware_mesh_refine
from MVMeshRecon.CoarseMeshRecon.CoarseRecon import CoarseRecon
# from MVMeshRecon.utils.general_utils import w_n2c_n
from tqdm import tqdm

from MVMeshRecon.refine_texture.api import opt_warpper
from pytorch3d.io import save_obj, load_objs_as_meshes
import pymeshlab as pml
from typing import List, Tuple, Optional

import pymeshlab as pml
def remesh(mesh):
    verts, faces = mesh.vertices, mesh.faces
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

    ms.apply_filter('meshing_isotropic_explicit_remeshing', targetlen=pml.PureValue(mean_edge_len), featuredeg=179)

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def resize_int_tensor(tensor, target_size):
    H, W = target_size
    tensor = torchvision.transforms.functional.resize(tensor.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    tensor = tensor.squeeze(0)
    return tensor

def cluster_points(points, method='hdbscan', **kwargs):
    """
    Cluster 3D points using either DBSCAN or HDBSCAN.

    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D points to cluster
    method : str, optional (default='hdbscan')
        Clustering method to use: 'dbscan' or 'hdbscan'
    **kwargs :
        Additional parameters for the clustering algorithm

        For DBSCAN:
            eps : float, optional (default=0.5)
                Maximum distance between points to be considered neighbors
            min_samples : int, optional (default=5)
                Minimum number of points to form a core point

        For HDBSCAN:
            min_cluster_size : int, optional (default=5)
                Minimum size of clusters
            min_samples : int, optional (default=None)
                Minimum number of points to form high-density cores

    Returns:
    --------
    list of numpy.ndarray
        List where each item is an array containing points belonging to the same cluster.
        The length of the list equals the number of valid clusters found.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if points.shape[1] != 3:
        raise ValueError("Points should be a (N,3) array for 3D clustering")

    # Set default parameters based on method
    if method.lower() == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)

        # Run DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)

    elif method.lower() == 'hdbscan':
        min_cluster_size = kwargs.get('min_cluster_size', 5)
        min_samples = kwargs.get('min_samples', None)

        # Run HDBSCAN clustering
        clustering = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

    else:
        raise ValueError("Method must be either 'dbscan' or 'hdbscan'")

        # Perform clustering
    labels = clustering.fit_predict(points)

    # Extract clusters and their indices, excluding noise points (labeled as -1)
    unique_labels = sorted(set(labels))
    clusters = []
    indices = []

    for label in unique_labels:
        # Skip noise points
        if label == -1:
            continue

        # Get mask for this cluster
        mask = labels == label

        # Get points for this cluster
        cluster_points = points[mask]

        # Get original indices for this cluster
        cluster_indices = np.where(mask)[0]

        clusters.append(cluster_points)
        indices.append(cluster_indices)

    return clusters, indices

def find_largest_connected_region(mask):
    """
    Find the largest connected component in a boolean mask and retain only that region.

    Parameters:
    mask (numpy.ndarray): Boolean array of shape [h, w]

    Returns:
    numpy.ndarray: Boolean array of same shape with only the largest connected component
    """
    # Label connected components
    labeled_array, num_features = ndimage.label(mask)

    if num_features == 0:
        return mask  # No True regions found

    # Count the number of pixels in each labeled region
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)

    # Find the label of the largest component
    largest_component_label = np.argmax(component_sizes) + 1  # +1 because we skipped background

    # Create a new mask with only the largest component
    largest_component_mask = (labeled_array == largest_component_label)

    return largest_component_mask

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000, device="cuda"):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).to(device), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'depth' in data:
            data['depth'] = torch.index_select(model_input['depth'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def get_max_component_mesh(mesh):

    connected_components = mesh.split(only_watertight=False)
    max_vertices = 0
    largest_component = None
    for component in connected_components:
        if len(component.vertices) > max_vertices:
            max_vertices = len(component.vertices)
            largest_component = component

    return largest_component

def get_obj_bbox(mesh, z_floor, delta):

    x_min, x_max = mesh.vertices[:, 0].min() - delta, mesh.vertices[:, 0].max() + delta
    y_min, y_max = mesh.vertices[:, 1].min() - delta, mesh.vertices[:, 1].max() + delta
    z_min, z_max = z_floor, mesh.vertices[:, 2].max() + delta

    obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]

    return obj_bbox

def calculate_bbox_distance(bbox_A, bbox_B):
    min_A, max_A = bbox_A
    min_B, max_B = bbox_B
    
    dist = np.zeros(3)
    for i in range(3):
        if max_A[i] < min_B[i]:
            dist[i] = min_B[i] - max_A[i]
        elif max_B[i] < min_A[i]:
            dist[i] = min_A[i] - max_B[i]
        else:
            dist[i] = 0

    return np.linalg.norm(dist)

def get_filtered_mesh(mesh, obj_bbox, bbox_dist_threshold):

    connected_components = mesh.split(only_watertight=False)
    filtered_connected_components = []

    for component in connected_components:

        x_mean, y_mean, z_mean = component.vertices.mean(axis=0)
        # mean points whether in the obj_bbox
        if obj_bbox[0][0] < x_mean < obj_bbox[1][0] and obj_bbox[0][1] < y_mean < obj_bbox[1][1] and obj_bbox[0][2] < z_mean < obj_bbox[1][2]:
            filtered_connected_components.append(component)
        else:           # if center not in obj bbox, check whether two bbox are near
            x_min, x_max = component.vertices[:, 0].min(), component.vertices[:, 0].max()
            y_min, y_max = component.vertices[:, 1].min(), component.vertices[:, 1].max()
            z_min, z_max = component.vertices[:, 2].min(), component.vertices[:, 2].max()
            obj_bbox_temp = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            # calculate two bbox distance
            bbox_dist = calculate_bbox_distance(obj_bbox, obj_bbox_temp)
            if bbox_dist < bbox_dist_threshold:
                filtered_connected_components.append(component)

    filtered_mesh = trimesh.util.concatenate(filtered_connected_components)
    return filtered_mesh


def render_textured_mesh(
        barycentric_coords,  # (H, W, 3)
        triangle_ids,  # (H, W)
        mask,  # (H, W)
        vt,  # (V', 2) texture vertices
        ft,  # (F', 3) texture faces
        texture_map,  # (H', W', 3) texture image
):
    # Get image dimensions
    H, W = mask.shape

    # Initialize output RGB image with zeros
    rgb = torch.zeros(H, W, 3, device=barycentric_coords.device)

    # Get the valid pixels (where mask is True)
    valid_pixels = torch.where(mask)

    if valid_pixels[0].numel() == 0:
        return rgb  # No valid pixels to render

    # Get triangle IDs and barycentric coordinates for valid pixels
    valid_triangle_ids = triangle_ids[valid_pixels]
    valid_bary_coords = barycentric_coords[valid_pixels]

    # Get texture face indices for each valid pixel
    texture_faces = ft[valid_triangle_ids]  # (N, 3)

    # Get texture vertex indices
    texture_vertex_indices = texture_faces

    # Get texture coordinates (u,v) for each vertex of each face
    texture_uvs = vt[texture_vertex_indices]  # (N, 3, 2)

    # Use barycentric coordinates to interpolate texture coordinates
    bary_coords_expanded = valid_bary_coords.unsqueeze(-1)  # (N, 3, 1)

    # Interpolate texture coordinates using barycentric weights
    interpolated_uvs = (texture_uvs * bary_coords_expanded).sum(dim=1)  # (N, 2)

    # Scale to texture image coordinates
    H_tex, W_tex = texture_map.shape[0], texture_map.shape[1]

    # Method 2: Implement bilinear interpolation manually
    u = interpolated_uvs[:, 0] * (W_tex - 1)
    v = interpolated_uvs[:, 1] * (H_tex - 1)

    # Get the four neighboring pixels
    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1 = torch.min(u0 + 1, torch.tensor(W_tex - 1, device=u.device))
    v1 = torch.min(v0 + 1, torch.tensor(H_tex - 1, device=v.device))

    # Calculate interpolation weights
    w_u1 = u - u0.float()
    w_v1 = v - v0.float()
    w_u0 = 1 - w_u1
    w_v0 = 1 - w_v1

    # Sample colors at the four corners
    c00 = texture_map[v0, u0]  # (N, 3)
    c01 = texture_map[v0, u1]  # (N, 3)
    c10 = texture_map[v1, u0]  # (N, 3)
    c11 = texture_map[v1, u1]  # (N, 3)

    # Perform bilinear interpolation
    c0 = w_u0.unsqueeze(-1) * c00 + w_u1.unsqueeze(-1) * c01  # Interpolate top row
    c1 = w_u0.unsqueeze(-1) * c10 + w_u1.unsqueeze(-1) * c11  # Interpolate bottom row
    c = w_v0.unsqueeze(-1) * c0 + w_v1.unsqueeze(-1) * c1  # Interpolate between rows

    # Assign interpolated colors to valid pixels in output image
    rgb[valid_pixels] = c

    return rgb

def rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w=None):

    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)


    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w[:3, :4], torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(vertices.device) @ vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = vert_cam[..., -1:]
        # depth_inverse = 1 / (depth + 1e-20)
        # depth_inverse, _ = dr.interpolate(depth_inverse.unsqueeze(0).contiguous(), rast_out, pos_idx)
        # depth = 1 / (depth_inverse + 1e-20)
        depth, _ = dr.interpolate(depth.unsqueeze(0).contiguous(), rast_out, pos_idx)
        depth = depth.reshape(H, W)


    return valid, triangle_id, depth

def rasterize_mesh_with_uv(mesh_dict, projection, glctx, resolution, c2w=None):

    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']
    vt = mesh_dict['vt']
    ft = mesh_dict['ft']
    texture_map = mesh_dict['texture_map']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    bary_coords = rast_out[..., :2].reshape(H, W, 2)
    bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)

    rgb = render_textured_mesh(
        bary_coords, triangle_id, valid, vt, ft, texture_map)

    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w[:3, :4], torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(vertices.device) @ vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = vert_cam[..., -1:]
        # depth_inverse = 1 / (depth + 1e-20)
        # depth_inverse, _ = dr.interpolate(depth_inverse.unsqueeze(0).contiguous(), rast_out, pos_idx)
        # depth = 1 / (depth_inverse + 1e-20)
        depth, _ = dr.interpolate(depth.unsqueeze(0).contiguous(), rast_out, pos_idx)
        depth = depth.reshape(H, W)


    return valid, triangle_id, depth, rgb

def rasterize_trimesh(mesh, projection, glctx, resolution, c2w=None):
    mesh_dict = {
        'vertices': F.pad(
            torch.from_numpy(mesh.vertices).float().to("cuda").contiguous(),
            pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(mesh.faces).int().to("cuda").contiguous()
    }

    return rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w)

def rasterize_mesh_dict_list(mesh_dict_list, projection, glctx, resolution, c2w=None):
    faces_cnts = [0]
    all_objs_meshes = []

    for mesh_dict in mesh_dict_list:
        mesh = trimesh.Trimesh(
            mesh_dict['vertices'][:, :3].cpu().numpy(),
            mesh_dict['pos_idx'].cpu().numpy(), process=False)

        all_objs_meshes.append(mesh)
        faces_cnts.append(faces_cnts[-1] + mesh.faces.shape[0])

    all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)
    mesh_dict = {
        'vertices': F.pad(
            torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
            pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
    }

    valid, triangle_id, depth = rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w)
    instance_id = torch.zeros_like(triangle_id)
    for mesh_i in range(len(faces_cnts) - 1):
        instance_id[torch.logical_and(triangle_id >= faces_cnts[mesh_i], triangle_id < faces_cnts[mesh_i + 1])] = mesh_i

    # instance_id[torch.logical_not(valid)] = -1

    return valid, triangle_id, depth, instance_id

def rasterize_mesh_dict_list_with_uv(mesh_dict_list, projection, glctx, resolution, c2w=None):
    valid, _, _, instance_id = rasterize_mesh_dict_list(mesh_dict_list, projection, glctx, resolution, c2w)

    H, W = resolution
    rgb = torch.zeros(H, W, 3).cuda()
    for mesh_i, mesh_dict in enumerate(mesh_dict_list):
        _, _, _, rgb_obj_i = rasterize_mesh_with_uv(mesh_dict, projection, glctx, resolution, c2w)
        rgb[instance_id == mesh_i] = rgb_obj_i[instance_id == mesh_i]

    return valid, instance_id, rgb


def read_obj_file(file_path):
    vertices = []
    texcoords = []
    faces_v = []  # Face vertex indices
    faces_vt = []  # Face texture coordinate indices

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            # Parse vertex coordinates
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            # Parse texture coordinates
            elif parts[0] == 'vt':
                texcoords.append([float(parts[1]), float(parts[2])])

            # Parse face indices
            elif parts[0] == 'f':
                face_v = []
                face_vt = []

                for part in parts[1:]:
                    indices = part.split('/')

                    # Vertex index (required) - convert from 1-based to 0-based indexing
                    if indices[0]:
                        face_v.append(int(indices[0]) - 1)

                    # Texture coordinate index (if present)
                    if len(indices) > 1 and indices[1]:
                        face_vt.append(int(indices[1]) - 1)
                    else:
                        face_vt.append(-1)  # Use -1 as placeholder if no texture index

                faces_v.append(face_v)
                faces_vt.append(face_vt)

    # Convert lists to NumPy arrays
    v = np.array(vertices, dtype=np.float32)
    vt = np.array(texcoords, dtype=np.float32)
    f = np.array(faces_v, dtype=np.int32)
    ft = np.array(faces_vt, dtype=np.int32)

    return v, f, vt, ft

def rasterize_mesh_list(mesh_list, projection, glctx, resolution, c2w=None):
    faces_cnts = [0]
    all_objs_meshes = []

    for mesh in mesh_list:
        all_objs_meshes.append(mesh)
        faces_cnts.append(faces_cnts[-1] + mesh.faces.shape[0])

    all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)
    mesh_dict = {
        'vertices': F.pad(
            torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
            pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
    }

    valid, triangle_id, depth = rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w)
    instance_id = torch.zeros_like(triangle_id)
    for mesh_i in range(len(faces_cnts) - 1):
        instance_id[torch.logical_and(triangle_id >= faces_cnts[mesh_i], triangle_id < faces_cnts[mesh_i + 1])] = mesh_i

    # instance_id[torch.logical_not(valid)] = -1

    return valid, triangle_id, depth, instance_id

def rasterize_mesh_list_front_face(mesh_list, projection, glctx, resolution, c2w=None):
    faces_cnts = [0]
    all_objs_meshes = []

    for mesh in mesh_list:
        all_objs_meshes.append(mesh)
        faces_cnts.append(faces_cnts[-1] + mesh.faces.shape[0])

    all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)

    face_normals = get_faces_normal(all_objs_mesh.vertices, all_objs_mesh.faces)
    face_normals = torch.from_numpy(face_normals).cuda().float()

    mesh_dict = {
        'vertices': F.pad(
            torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
            pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
    }

    valid, triangle_id, depth = rasterize_mesh(mesh_dict, projection, glctx, resolution, c2w)
    cam_normals = get_cam_normal_from_rast(valid, triangle_id, face_normals, c2w)
    front_face = cam_normals[..., 2] < 0
    valid = torch.logical_and(valid, front_face)

    instance_id = torch.zeros_like(triangle_id)
    for mesh_i in range(len(faces_cnts) - 1):
        instance_id[torch.logical_and(triangle_id >= faces_cnts[mesh_i], triangle_id < faces_cnts[mesh_i + 1])] = mesh_i

    # instance_id[torch.logical_not(valid)] = -1

    return valid, triangle_id, depth, instance_id


def total_variation_loss(normal_map, mask, weight=1.0):
    """TV loss implementation that respects object boundaries"""
    # Similar to above, but using L2 norm instead of L1
    diff_x = torch.pow(normal_map[:, :, :, 1:] - normal_map[:, :, :, :-1], 2)
    diff_y = torch.pow(normal_map[:, :, 1:, :] - normal_map[:, :, :-1, :], 2)

    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    # L2 norm
    if mask_x.sum() == 0:
        loss_x = torch.tensor(0.0).cuda().float()
    else:
        loss_x = torch.sqrt((diff_x * mask_x).sum(dim=1) + 1e-8).sum() / (mask_x.sum() + 1e-8)
    if mask_y.sum() == 0:
        loss_y = torch.tensor(0.0).cuda().float()
    else:
        loss_y = torch.sqrt((diff_y * mask_y).sum(dim=1) + 1e-8).sum() / (mask_y.sum() + 1e-8)

    return weight * (loss_x + loss_y)

def get_normal_map_from_depth(depth, mask, scale):
    H, W = depth.shape[:2]
    near = 0.001
    ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near, torch.eye(4), scale)
    ray_origins = ray_origins.cpu().numpy().reshape(H, W, 3)
    ray_dirs = ray_dirs.cpu().numpy().reshape(H, W, 3)

    ray_origins = ray_origins[mask].reshape(-1, 3)
    ray_dirs = ray_dirs[mask].reshape(-1, 3)
    depth = depth[mask].reshape(-1, 1)

    points = ray_origins + ray_dirs * depth
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    pcd.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))

    normals = np.array(pcd.normals).reshape(-1, 3)
    normals[normals[:, -1] > 0, :] *= -1

    normal_map = np.ones((H, W, 3))
    normal_map[mask] = normals

    return normal_map

def second_order_smoothness(normal_map, mask, weight=1.0):


    """Penalize second derivatives for stronger smoothness"""
    padded = F.pad(normal_map, (1, 1, 1, 1), mode='replicate')
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode='replicate')

    # Compute Laplacian (second derivative approximation)
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=normal_map.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # Apply laplacian separately to each channel
    lap_x = F.conv2d(padded[:, 0:1], laplacian_kernel[:1], padding=0)
    lap_y = F.conv2d(padded[:, 1:2], laplacian_kernel[1:2], padding=0)
    lap_z = F.conv2d(padded[:, 2:3], laplacian_kernel[2:3], padding=0)

    laplacian = torch.cat([lap_x, lap_y, lap_z], dim=1)

    # Apply mask
    valid_mask = padded_mask[:, :, 1:-1, 1:-1]

    if valid_mask.sum() == 0:
        return torch.tensor(0.0).cuda().float()

    loss = torch.abs(laplacian * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    return weight * loss


def load_tex_dict_from_tex_mesh_p3d(mesh_p3d):
    tex_dict = {}

    verts = mesh_p3d.verts_packed().reshape(-1, 3).cpu().numpy()
    faces = mesh_p3d.faces_packed().reshape(-1, 3).cpu().numpy()
    vts = mesh_p3d.textures.verts_uvs_padded().reshape(-1, 2).cpu().numpy()
    vts[:, 1] = 1 - vts[:, 1]  # Flip y-coordinates
    fts = mesh_p3d.textures.faces_uvs_padded().reshape(-1, 3).cpu().numpy()
    tex_map = mesh_p3d.textures.maps_padded()[0].cpu().numpy()

    tex_dict['vertices'] = verts
    tex_dict['faces'] = faces
    tex_dict['vt'] = vts
    tex_dict['ft'] = fts
    tex_dict['texture_map'] = tex_map

    return tex_dict

def load_tex_dict_from_path(tex_mesh_path):

    mesh_p3d = load_objs_as_meshes([tex_mesh_path])
    tex_dict = load_tex_dict_from_tex_mesh_p3d(mesh_p3d)
    tex_dict = get_tex_mesh_dict_for_nvrast(tex_dict)
    return tex_dict

def get_tex_mesh_dict_for_nvrast(tex_dict_np):
    tex_dict = {
        'vertices': F.pad(
            torch.from_numpy(tex_dict_np['vertices']).float().to("cuda").contiguous(),
            pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(tex_dict_np['faces']).int().to("cuda").contiguous(),
        'vt': torch.from_numpy(tex_dict_np['vt']).float().to("cuda").contiguous(),
        'ft': torch.from_numpy(tex_dict_np['ft']).int().to("cuda").contiguous(),
        'texture_map': torch.from_numpy(tex_dict_np['texture_map']).float().to("cuda").contiguous()
    }
    return tex_dict


def rasterize_mesh_vert_colors(mesh_dict, projection, glctx, resolution):
    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']
    vert_colors = mesh_dict['vert_colors']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)

    triangle_id = (rast_out[..., -1] - 1).long().reshape(-1)
    barys_0 = rast_out[..., 0].float().reshape(-1, 1, 1)
    barys_1 = rast_out[..., 1].float().reshape(-1, 1, 1)
    barys_2 = 1 - barys_0 - barys_1

    triangle_colors = vert_colors[pos_idx.reshape(-1).long()].reshape(-1, 3, 3)
    triangle_colors = triangle_colors[triangle_id.reshape(-1)].reshape(-1, 3, 3)

    colors = triangle_colors[:, 0:1] * barys_0 + triangle_colors[:, 1:2] * barys_1 + triangle_colors[:, 2:3] * barys_2
    colors = torch.sum(colors, dim=1).reshape(H, W, 3)

    return colors, valid, triangle_id.reshape(H, W)


def rasterize_mesh_return_pixel_vert_and_bary(mesh_dict, projection, glctx, resolution):
    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']
    vertices_world = mesh_dict['vertices_world']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)

    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)
    barys_0 = rast_out[..., 0].float().reshape(H, W)
    barys_1 = rast_out[..., 1].float().reshape(H, W)
    barys_2 = 1 - barys_0 - barys_1
    barys = torch.stack([barys_0, barys_1, barys_2], dim=-1)

    vertices_id = pos_idx[triangle_id.reshape(-1)].reshape(H, W, 3)
    vertices_world = vertices_world[vertices_id.reshape(-1)].reshape(H, W, 3, 3)

    return valid, triangle_id, vertices_world, barys

def rasterize_mesh_depth_peeler(mesh_dict, projection, glctx, resolution):
    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    # rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)
    H, W = resolution

    valid_list = []
    triangle_id_list = []

    with dr.DepthPeeler(glctx, vertices_clip, pos_idx, resolution) as peeler:
        for peel_i in range(3):
            rast_out, _ = peeler.rasterize_next_layer()

            valid = (rast_out[..., -1] > 0).reshape(H, W)
            triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

            valid_list.append(valid)
            triangle_id_list.append(triangle_id)

    return valid_list, triangle_id_list


def get_cam_normal_from_rast(valid, triangle_id, face_normals, pose):
    H, W = valid.shape
    face_normals = face_normals[triangle_id.reshape(-1)].reshape(-1, 3)
    rot_w2c = pose[:3, :3].T
    cam_face_normals = ((rot_w2c @ face_normals.T).T).reshape(H, W, 3)
    cam_face_normals = F.normalize(cam_face_normals, dim=-1)
    return cam_face_normals

def fov_to_focal_length(fov, resolution):
    H, W = resolution
    fx = W / (2 * np.tan(fov / 2))
    fy = H / (2 * np.tan(fov / 2))
    return fx, fy

def get_camera_perspective_projection_matrix(fx, fy, cx, cy, H, W, near, far):
    projection = np.zeros((4, 4))
    projection[0, 0] = 2.0 * fx / W
    projection[1, 1] = 2.0 * fy / H
    projection[0, 2] = 2.0 * (cx / W - 0.5)
    projection[1, 2] = 2.0 * (cy / H - 0.5)
    projection[2, 2] = (far + near) / (far - near)
    projection[2, 3] = -2.0 * far * near / (far - near)
    projection[3, 2] = 1.0

    return projection.astype(np.float32)

def get_camera_perspective_rays(fx, fy, cx, cy, H, W):
    pixel_x = np.arange(W)
    pixel_y = np.arange(H)
    pixel_x, pixel_y = np.meshgrid(pixel_x, pixel_y, indexing='ij')
    pixel_xy = np.stack([pixel_x, pixel_y], axis=-1)
    pixels = pixel_xy.transpose((1, 0, 2)) # (H, W, 2)

    ray_origins = np.zeros((H, W, 3))
    ray_origins[..., 0] = (pixels[..., 0] - cx) / fx
    ray_origins[..., 1] = (pixels[..., 1] - cy) / fy
    ray_origins[..., 2] = 1.0

    rays_direction = ray_origins / np.linalg.norm(ray_origins, axis=-1, keepdims=True)

    ray_origins = np.zeros((H, W, 3))

    return ray_origins, rays_direction

def get_camera_perspective_rays_world(fx, fy, cx, cy, H, W, pose):
    ray_origins, rays_direction = get_camera_perspective_rays(fx, fy, cx, cy, H, W)
    c2w = pose
    rot = pose[:3, :3]

    ray_origins = ray_origins.reshape(-1, 3)
    ray_origins_pad_cam = np.concatenate([ray_origins, np.ones((H * W, 1))], axis=-1)
    ray_origins_pad_world = (c2w @ ray_origins_pad_cam.T).T
    ray_origins = ray_origins_pad_world[:, :3] / ray_origins_pad_world[:, 3:4]

    rays_direction = (rot @ rays_direction.reshape(-1, 3).T).T

    rays_direction = rays_direction / np.linalg.norm(rays_direction, axis=-1, keepdims=True)

    return ray_origins.reshape(H, W, 3).astype(np.float32), rays_direction.reshape(H, W, 3).astype(np.float32)

def get_camera_orthogonal_projection_matrix(near, far, scale=1.0):
    projection = np.eye(4)
    projection[2, 2] = 2.0 / (far - near)
    projection[2, 3] = -(far + near) / (far - near)

    projection[0, 0] = (1 / scale)
    projection[1, 1] = (1 / scale)

    return projection.astype(np.float32)

def get_camera_orthogonal_projection_matrix_offset(near, far, height, width, scale=1.0, offsets=None):
    projection = np.eye(4)
    projection[2, 2] = 2.0 / (far - near)
    projection[2, 3] = -(far + near) / (far - near)

    # Basic scale factors
    projection[0, 0] = (1 / scale)
    projection[1, 1] = (1 / scale)

    # Generate random offsets within ±0.5 pixels
    # Convert to normalized device coordinates by dividing by dimensions
    if offsets is not None:
        x_offset = offsets[0]
        y_offset = offsets[1]
    else:
        x_offset = (np.random.random() - 0.5) / width
        y_offset = (np.random.random() - 0.5) / height

    # Apply the offsets to the translation components
    projection[0, 3] = x_offset * 2.0  # *2 to convert from [0,1] to [-1,1] NDC space
    projection[1, 3] = y_offset * 2.0

    return projection.astype(np.float32)

def get_camera_orthogonal_rays(H, W, near, pose, scale=1.0):
    pixel_x = np.arange(W)
    pixel_y = np.arange(H)
    pixel_x, pixel_y = np.meshgrid(pixel_x, pixel_y, indexing='ij')
    pixel_xy = np.stack([pixel_x, pixel_y], axis=-1)
    pixels = pixel_xy.transpose((1, 0, 2)) # (H, W, 2)

    ray_origins_pad_cam = np.concatenate([
        (pixels[..., 0:1] * 2 / W - 1) * scale,
        (pixels[..., 1:2] * 2 / H - 1) * scale,
        np.ones((H, W, 1)) * near,
        np.ones((H, W, 1))
        ], axis=-1)

    ray_origins_pad_cam = torch.from_numpy(ray_origins_pad_cam).float().reshape(-1, 4)

    rays_direction = np.concatenate([
        np.zeros((H, W, 2)),
        np.ones((H, W, 1))
        ], axis=-1)

    rays_direction = torch.from_numpy(rays_direction).float().reshape(-1, 3)

    pose = pose[:3, :4]
    rot = pose[:3, :3]

    ray_origins = (pose @ ray_origins_pad_cam.T).T
    rays_direction = (rot @ rays_direction.T).T

    rays_direction = torch.nn.functional.normalize(rays_direction, dim=-1)

    return ray_origins, rays_direction


def visualize_view_weights(view_weights, output_path='view_weights_visualization.png', dpi=300, altitude_range=(60, 120)):
    """
    Visualize view weights on a spherical projection and save to a PNG file.

    Parameters:
    view_weights: numpy array of shape (num_azimuth_bins, num_altitude_bins)
    output_path: path to save the PNG file (default: 'view_weights_visualization.png')
    dpi: resolution of the output image (default: 300)
    """
    # Get dimensions
    num_azimuth_bins, num_altitude_bins = view_weights.shape

    # Create figure with Mollweide projection (good for spherical visualization)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='mollweide')

    # Create a meshgrid for azimuth and altitude
    # Convert bin indices to radians for Mollweide projection
    phi = np.linspace(-np.pi, np.pi, num_azimuth_bins)
    theta = np.linspace((altitude_range[1] - 90.0) * np.pi / 180.0, (altitude_range[0] - 90.0) * np.pi / 180.0, num_altitude_bins)

    # Create 2D arrays for pcolormesh
    Phi, Theta = np.meshgrid(phi, theta)

    # We need to properly orient the data for display
    # For Mollweide projection, we need the data in the right shape
    weights_display = view_weights

    # Create custom orange-red colormap
    colors = [(1, 1, 1, 0),  # White (transparent) for low values
              (1, 0.6, 0.2, 0.5),  # Light orange for medium values
              (1, 0.3, 0, 1)]  # Dark orange/red for high values
    orange_cmap = LinearSegmentedColormap.from_list('OrangeRed', colors)

    # Plot the data with correct orientation
    # In Mollweide projection, we need to be careful about the orientation
    # We need to transpose and flip appropriately
    mesh = ax.pcolormesh(Phi, Theta, weights_display.T, cmap=orange_cmap, shading='auto')

    # Add grid lines
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')

    # Set azimuth ticks (convert to degrees for display)
    azimuth_ticks = np.array([30-180, 120-180, 180-180, 240-180, 330-180]) * np.pi / 180
    ax.set_xticks(azimuth_ticks)
    ax.set_xticklabels(['$30°$', '$120°$', '$180°$', '$240°$', '$330°$'])

    # Remove y-axis ticks to match the reference image
    ax.set_yticks([])

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Create a thin horizontal line at the equator (theta=0)
    ax.axhline(y=0, color='black', alpha=0.7, linewidth=1.5)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Tight layout
    plt.tight_layout()

    # Save figure to file
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)

    return output_path


def visualize_view_weights_points(view_weights, output_path='view_weights_visualization.png', dpi=300,
                           altitude_range=(60, 120), picked_views=None):
    """
    Visualize view weights on a spherical projection and save to a PNG file.
    Optionally show picked representative views.

    Parameters:
    view_weights: numpy array of shape (num_azimuth_bins, num_altitude_bins)
    output_path: path to save the PNG file (default: 'view_weights_visualization.png')
    dpi: resolution of the output image (default: 300)
    altitude_range: tuple of min and max altitude in degrees (default: (40, 140))
    picked_views: list of tuples (azimuth, altitude) showing representative views
    """
    # Get dimensions
    num_azimuth_bins, num_altitude_bins = view_weights.shape

    # Create figure with Mollweide projection (good for spherical visualization)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='mollweide')

    # Create a meshgrid for azimuth and altitude
    # Convert bin indices to radians for Mollweide projection
    phi = np.linspace(-np.pi, np.pi, num_azimuth_bins)
    theta = np.linspace((altitude_range[1] - 90.0) * np.pi / 180.0, (altitude_range[0] - 90.0) * np.pi / 180.0,
                        num_altitude_bins)

    # Create 2D arrays for pcolormesh
    Phi, Theta = np.meshgrid(phi, theta)

    # We need to properly orient the data for display
    # For Mollweide projection, we need the data in the right shape
    weights_display = view_weights

    # Create custom orange-red colormap
    colors = [(1, 1, 1, 0),  # White (transparent) for low values
              (1, 0.6, 0.2, 0.5),  # Light orange for medium values
              (1, 0.3, 0, 1)]  # Dark orange/red for high values
    orange_cmap = LinearSegmentedColormap.from_list('OrangeRed', colors)

    # Plot the data with correct orientation
    # In Mollweide projection, we need to be careful about the orientation
    # We need to transpose and flip appropriately
    mesh = ax.pcolormesh(Phi, Theta, weights_display.T, cmap=orange_cmap, shading='auto')

    # Plot representative views if provided
    if picked_views is not None:
        for azi, alt in picked_views:
            # Convert to radians and adjust for Mollweide projection
            azi_rad = (azi - 180) * np.pi / 180  # Convert and center at 0
            alt_rad = -(alt - 90) * np.pi / 180  # Convert from 0-180 to -90-90

            # Plot a marker for this view
            ax.plot(azi_rad, alt_rad, 'o',
                    color='blue', markersize=8, markeredgecolor='white', markeredgewidth=1.5)

            # Add text label showing (azimuth, altitude)
            ax.text(azi_rad, alt_rad, f'({int(azi)}°, {int(alt)}°)',
                    color='white', fontsize=8, ha='center', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # Add grid lines
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')

    # Set azimuth ticks (convert to degrees for display)
    azimuth_ticks = np.array([30 - 180, 120 - 180, 180 - 180, 240 - 180, 330 - 180]) * np.pi / 180
    ax.set_xticks(azimuth_ticks)
    ax.set_xticklabels(['$30°$', '$120°$', '$180°$', '$240°$', '$330°$'])

    # Remove y-axis ticks to match the reference image
    ax.set_yticks([])

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Create a thin horizontal line at the equator (theta=0)
    ax.axhline(y=0, color='black', alpha=0.7, linewidth=1.5)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add title with number of picked views (if any)
    if picked_views is not None:
        plt.title(f'View Weight Distribution with {len(picked_views)} Representative Views', fontsize=12)

    # Tight layout
    plt.tight_layout()

    # Save figure to file
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)

    return output_path
def get_scale_shift(verts):
    verts_bounds = np.array([verts.min(axis=0), verts.max(axis=0)])
    shift = (verts_bounds[0] + verts_bounds[1]) / 2
    scale = np.max(verts_bounds[1] - verts_bounds[0]) * (2/3)
    return scale, shift.reshape(1, 3)

def apply_scale_shift(p, scale, shift):
    p_shape = p.shape
    return ((p.reshape(-1, 3) - shift) / scale).reshape(p_shape)

def apply_inv_scale_shift(p, scale, shift):
    p_shape = p.shape
    p = p.reshape(-1, 3)
    p = p * scale + shift
    return p.reshape(p_shape)


def find_contiguous_in_array(p):
    """
    Given a 1D boolean numpy array p (shape (N,)) treated circularly,
    find contiguous blocks of True values. For each block of length L:

      - If L == 1, add that index.
      - If L == 2, add both indices.
      - If L >= 3 and is even, add the two middle indices.
      - If L >= 3 and is odd, add the three middle indices.

    Returns a list of these 'middle' indices for all blocks.
    """
    N = len(p)
    outputs = []

    # Quick check: if everything is False, return empty.
    if not np.any(p):
        return outputs  # []

    visited = np.zeros(N, dtype=bool)

    for i in range(N):
        # If we haven't visited i yet and p[i] is True, we found a new block start
        if p[i] and not visited[i]:
            block_indices = [i]
            visited[i] = True

            j = i
            # Walk forward in circular manner as long as we see True and not visited
            while True:
                nxt = (j + 1) % N
                if p[nxt] and not visited[nxt]:
                    block_indices.append(nxt)
                    visited[nxt] = True
                    j = nxt
                else:
                    break

            L = len(block_indices)

            # Decide how to pick the "middle" indices based on the block length
            if L == 1:
                # Single element
                outputs.append(block_indices[0])
            elif L == 2:
                # Just take both
                outputs.extend(block_indices)
            else:
                # L >= 3
                if L % 2 == 0:
                    # Even length: add the two central indices
                    mid1 = L // 2 - 1
                    mid2 = L // 2
                    outputs.append(block_indices[mid1])
                    outputs.append(block_indices[mid2])
                else:
                    # Odd length: add the three central indices
                    c = L // 2
                    # For L = 3, c = 1 => block_indices[0..2]
                    # For L = 5, c = 2 => block_indices[1..3], etc.
                    start_idx = c - 1
                    end_idx = c + 2
                    outputs.extend(block_indices[start_idx:end_idx])

    return outputs

def highest_sampling_view_weights(view_weights, num_pick_views=6, seed=None):
    num_azimuth_bins, num_altitude_bins = view_weights.shape
    azi_values = np.linspace(0, 360, num_azimuth_bins, endpoint=False)
    alt_values = np.linspace(60, 120, num_altitude_bins)

    select_idxs = np.argsort(view_weights.reshape(-1))[-num_pick_views:]
    selected_views = [(azi_values[idx // num_altitude_bins], alt_values[idx % num_altitude_bins]) for idx in select_idxs]
    return selected_views

def margin_aware_fps_sampling(view_weights, num_pick_views=6, neighborhood_size=6, seed=None):
    """
    Sample representative views using FPS that avoids margins near low-visibility regions.

    Parameters:
    view_weights: numpy array of shape (num_azimuth_bins, num_altitude_bins)
    num_pick_views: number of views to select
    neighborhood_size: size of neighborhood to check for margin detection
    seed: random seed for reproducibility

    Returns:
    list of tuples (azimuth, altitude) representing selected views
    """
    if seed is not None:
        np.random.seed(seed)

    num_azimuth_bins, num_altitude_bins = view_weights.shape

    # Create coordinate arrays
    azi_values = np.linspace(0, 360, num_azimuth_bins, endpoint=False)
    alt_values = np.linspace(60, 120, num_altitude_bins)

    # Create mesh grid of coordinates
    azi_grid, alt_grid = np.meshgrid(azi_values, alt_values)
    azi_flat = azi_grid.flatten()
    alt_flat = alt_grid.flatten()

    # Normalize weights
    weights_norm = view_weights / np.max(view_weights)

    # Calculate margin scores for each point
    margin_scores = np.zeros_like(weights_norm)
    for i in range(num_azimuth_bins):
        for j in range(num_altitude_bins):
            # Define neighborhood indices with wraparound for azimuth
            azi_indices = [(i + k) % num_azimuth_bins for k in range(-neighborhood_size, neighborhood_size + 1)]
            alt_indices = [max(0, min(num_altitude_bins - 1, j + k)) for k in
                           range(-neighborhood_size, neighborhood_size + 1)]

            # Get neighborhood values
            neighborhood = [weights_norm[a, b] for a in azi_indices for b in alt_indices]

            # Calculate variance and min value in neighborhood
            variance = np.var(neighborhood)
            min_val = np.min(neighborhood)

            # High variance and low min value indicate a margin area
            margin_scores[i, j] = variance * (1 - min_val)

    # Normalize margin scores
    margin_scores = margin_scores / np.max(margin_scores) if np.max(margin_scores) > 0 else margin_scores

    # Penalize points in margin areas
    weights_enhanced = weights_norm ** 3  # Enhance contrast
    margin_penalty = 1 - margin_scores  # Convert to penalty factor
    safe_weights = weights_enhanced * margin_penalty

    # Apply threshold to completely avoid very low visibility areas
    threshold = 0.2 * np.max(safe_weights)
    safe_weights_flat = safe_weights.T.flatten()  # Transpose to match grid ordering
    safe_weights_flat[safe_weights_flat < threshold] = 0

    # Create probability distribution
    prob_dist = safe_weights_flat / np.sum(safe_weights_flat) if np.sum(safe_weights_flat) > 0 else None

    # Handle case where all weights might be zero
    if prob_dist is None:
        # Fall back to original weights if all safe weights are zero
        prob_dist = weights_norm.T.flatten()
        prob_dist = prob_dist / np.sum(prob_dist)

    # Initialize with a point sampled from the probability distribution
    selected_indices = [np.random.choice(len(azi_flat), p=prob_dist)]
    selected_views = [(azi_flat[selected_indices[0]], alt_flat[selected_indices[0]])]

    # Pre-compute all pairwise distances
    distances = np.zeros((len(azi_flat), len(azi_flat)))
    for i in range(len(azi_flat)):
        for j in range(i + 1, len(azi_flat)):
            # Calculate angular distance (handling azimuth wrap-around)
            azi_diff = min(abs(azi_flat[i] - azi_flat[j]), 360 - abs(azi_flat[i] - azi_flat[j]))
            alt_diff = abs(alt_flat[i] - alt_flat[j])
            angular_dist = np.sqrt(azi_diff ** 2 + alt_diff ** 2)
            distances[i, j] = angular_dist
            distances[j, i] = angular_dist

    # Iteratively select points based on FPS with margin and density awareness
    for _ in range(1, num_pick_views):
        # Calculate minimum distance to existing samples for each point
        min_dists = np.array([np.min([distances[i, j] for j in selected_indices]) for i in range(len(azi_flat))])

        # Combine distance, density, and margin awareness
        sampling_weights = min_dists * prob_dist

        # Avoid reselecting points
        for idx in selected_indices:
            sampling_weights[idx] = 0

        # Handle case where all weights might be zero
        if np.sum(sampling_weights) == 0:
            break

        # Normalize to create probability distribution
        sampling_weights = sampling_weights / np.sum(sampling_weights)

        # Select next point
        next_idx = np.random.choice(len(azi_flat), p=sampling_weights)
        selected_indices.append(next_idx)
        selected_views.append((azi_flat[next_idx], alt_flat[next_idx]))

    return selected_views

def uniform_metric(view_weights):
    """
    Measure how uniform the view weights are, with priority on azimuth uniformity.

    Returns a value between 0 and 1, where higher values indicate greater uniformity.
    """
    # Normalize weights
    weights_norm = view_weights / np.max(view_weights)

    # Calculate entropy-based uniformity (higher entropy = more uniform)
    epsilon = 1e-10  # Prevent log(0)

    # Azimuth uniformity (mean across altitudes for each azimuth)
    azi_means = np.max(weights_norm, axis=1)
    azi_means_norm = azi_means / np.sum(azi_means)
    azi_entropy = -np.sum(azi_means_norm * np.log(azi_means_norm + epsilon))
    azi_uniformity = azi_entropy / np.log(len(azi_means))  # Normalize by max entropy

    # Weighted combination (prioritizing azimuth)
    uniformity = 1.0 * azi_uniformity

    return uniformity


def add_view(old_view_weights, new_view_params):
    """
    Add a new viewing direction that provides more information to the existing view weights.

    Parameters:
    old_view_weights: numpy array of shape (num_azimuths, num_altitudes)
    new_view_params: tuple (azi_new, alt_new) or (azi_new, alt_new, spread_azi, spread_alt, strength)

    Returns:
    new_view_weights: updated view weights array
    """
    num_azimuths, num_altitudes = old_view_weights.shape

    # Extract parameters
    azi_new, alt_new = new_view_params[0], new_view_params[1]

    # Default parameters if not provided
    spread_azi = new_view_params[2] if len(new_view_params) > 2 else 90  # degrees
    spread_alt = new_view_params[3] if len(new_view_params) > 3 else 90  # degrees

    # Dynamically determine strength based on the existing view weight distribution
    if len(new_view_params) > 4:
        # Use provided strength if available
        strength = new_view_params[4]
    else:
        # Calculate adaptive strength based on the distribution characteristics
        max_weight = np.max(old_view_weights)
        mean_weight = np.mean(old_view_weights)

        # Measure distribution uniformity
        normalized_weights = old_view_weights / max_weight
        non_zero_weights = normalized_weights[normalized_weights > 0.05]

        if len(non_zero_weights) > 0:
            weight_std = np.std(non_zero_weights)
            coverage_ratio = len(non_zero_weights) / old_view_weights.size
        else:
            weight_std = 1.0
            coverage_ratio = 0.0

        # Higher strength when:
        # 1. Low coverage (many regions have little/no visibility)
        # 2. High variance in weights (very uneven distribution)
        # 3. Low mean value (overall poor visibility)
        base_strength = 0.75  # Baseline strength
        coverage_factor = 1.5 * (1 - coverage_ratio)  # More strength when coverage is poor
        uniformity_factor = weight_std  # More strength when distribution is uneven
        intensity_factor = 1 - (mean_weight / max_weight)  # More strength when overall weights are low

        # Combine factors (with caps to prevent extreme values)
        strength = base_strength * (1 +
                                    min(1.0, coverage_factor) +
                                    min(1.0, uniformity_factor) +
                                    min(1.0, intensity_factor))

        # Ensure strength is in a reasonable range
        strength = max(0.5, min(2.5, strength))

    # Create coordinate grid
    azi_range = np.linspace(0, 360, num_azimuths, endpoint=False)
    alt_range = np.linspace(60, 120, num_altitudes)
    azi_grid, alt_grid = np.meshgrid(azi_range, alt_range, indexing='ij')

    # Calculate angular distances (handling azimuth wrap-around)
    azi_diff = np.minimum(np.abs(azi_grid - azi_new), 360 - np.abs(azi_grid - azi_new))
    alt_diff = np.abs(alt_grid - alt_new)

    # Create visibility contribution using Gaussian
    gaussian = np.exp(-(azi_diff ** 2 / (2 * spread_azi ** 2) + alt_diff ** 2 / (2 * spread_alt ** 2)))

    # Scale gaussian by the inverse of existing weights
    # This makes the contribution greater in areas with low visibility
    # weight_scale = 1 - (old_view_weights / np.max(old_view_weights))
    weight_scale = 0.3
    contribution = strength * gaussian * (0.3 + 0.7 * weight_scale)

    # Combine with existing weights
    # Using a blend of maximum and addition approaches
    new_view_weights = np.maximum(
        old_view_weights,
        contribution * np.max(old_view_weights)
    )

    return new_view_weights


def find_best_additional_view(old_view_weights, sample_density=10):
    """
    Find the optimal additional view that maximizes uniformity.

    Parameters:
    old_view_weights: numpy array of shape (num_azimuths, num_altitudes)
    sample_density: controls how many points to sample in the search space

    Returns:
    tuple: (best_azi, best_alt, best_uniformity, best_new_weights)
    """
    # Sample the azimuth-altitude space
    num_azi_samples = 36 * sample_density  # Every 10/sample_density degrees
    num_alt_samples = 1 * sample_density  # 10*sample_density samples in 40-140° range

    azi_values = np.linspace(0, 360, num_azi_samples, endpoint=False)
    alt_values = np.linspace(85, 95, num_alt_samples)

    best_uniformity = -1
    best_azi, best_alt = None, None
    best_new_weights = None

    # Grid search for the best viewing direction
    for azi in azi_values:
        for alt in alt_values:
            # Try adding this view
            new_weights = add_view(old_view_weights, (azi, alt))

            # Evaluate uniformity
            uniformity = uniform_metric(new_weights)

            # Update if better
            if uniformity > best_uniformity:
                best_uniformity = uniformity
                best_azi, best_alt = azi, alt
                best_new_weights = new_weights

    return best_azi, best_alt, best_uniformity, best_new_weights


def evaluate_view_addition(old_view_weights, new_view_weights, uniformity_threshold=0.001):
    """
    Determine whether adding the new view is worthwhile.

    Parameters:
    old_view_weights: original view weights array
    new_view_weights: view weights after adding the new view
    uniformity_threshold: minimum improvement in uniformity required

    Returns:
    tuple: (should_add, metrics_dict)
    """
    # Calculate uniformity metrics
    old_uniformity = uniform_metric(old_view_weights)
    new_uniformity = uniform_metric(new_view_weights)
    uniformity_improvement = new_uniformity - old_uniformity

    should_add = uniformity_improvement > uniformity_threshold

    metrics = {
        'uniformity_improvement': uniformity_improvement,
        'should_add': should_add
    }

    return should_add, metrics

def find_longest_contiguous_in_array(p):
    """
    Given a 1D boolean numpy array p (shape (N,)) treated circularly,
    find contiguous blocks of True values. For each block of length L:

      - If L == 1, add that index.
      - If L == 2, add both indices.
      - If L >= 3 and is even, add the two middle indices.
      - If L >= 3 and is odd, add the three middle indices.

    Returns a list of these 'middle' indices for all blocks.
    """
    N = len(p)
    outputs = []
    longest_len = 0

    # Quick check: if everything is False, return empty.
    if not np.any(p):
        return outputs  # []

    visited = np.zeros(N, dtype=bool)

    for i in range(N):
        # If we haven't visited i yet and p[i] is True, we found a new block start
        if p[i] and not visited[i]:
            block_indices = [i]
            visited[i] = True

            j = i
            # Walk forward in circular manner as long as we see True and not visited
            while True:
                nxt = (j + 1) % N
                if p[nxt] and not visited[nxt]:
                    block_indices.append(nxt)
                    visited[nxt] = True
                    j = nxt
                else:
                    break

            L = len(block_indices)

            sampled_views_len = int(0.6 * L + 0.5)
            if sampled_views_len > longest_len:
                longest_len = sampled_views_len
                actual_sampled_views_len = min(sampled_views_len, 6)
                outputs = [elem for idx, elem in enumerate(block_indices) if idx in (L // 2 + np.arange(actual_sampled_views_len) - (actual_sampled_views_len // 2)).tolist()]

            # # Decide how to pick the "middle" indices based on the block length
            # if L == 1:
            #     # Single element
            #     if longest_len < 1:
            #         longest_len = 1
            #         outputs = block_indices
            #     # outputs.append(block_indices[0])
            # elif L == 2:
            #     # Just take both
            #     if longest_len < 2:
            #         longest_len = 2
            #         outputs = block_indices
            #     # outputs.extend(block_indices)
            # else:
            #     # L >= 3
            #     if L % 2 == 0:
            #         # Even length: add the two central indices
            #         mid1 = L // 2 - 1
            #         mid2 = L // 2
            #         if longest_len < L:
            #             longest_len = L
            #             outputs = [block_indices[mid1], block_indices[mid2]]
            #         # outputs.append(block_indices[mid1])
            #         # outputs.append(block_indices[mid2])
            #     else:
            #         # Odd length: add the three central indices
            #         c = L // 2
            #         # For L = 3, c = 1 => block_indices[0..2]
            #         # For L = 5, c = 2 => block_indices[1..3], etc.
            #         start_idx = c - 1
            #         end_idx = c + 2
            #         if longest_len < L:
            #             longest_len = L
            #             outputs = block_indices[start_idx:end_idx]
            #         # outputs.extend(block_indices[start_idx:end_idx])

    return outputs

def vis_prune(verts, faces, vert_colors, vis_colors, glctx):
    radius = 1.0
    near = 0.001
    far = 100.0

    camera_projmat = get_camera_orthogonal_projection_matrix(near, far)
    camera_projmat = torch.from_numpy(camera_projmat).float().cuda()

    mesh_dict = {
        'vertices': F.pad(verts.contiguous(),
                          pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': faces.contiguous(),
        'vert_colors': vert_colors
    }
    H, W = 256, 256

    num_thetas = 30
    thetas = np.linspace(0, 360.0, num=num_thetas, endpoint=False)
    # phis = np.linspace(-5.0 + 90.0, 5.0 + 90.0, num=3)
    phis = [90.0]

    faces_keep = torch.zeros(faces.shape[0], dtype=torch.bool)
    faces_keep[vis_colors > 0] = True

    for theta_i, theta in enumerate(thetas):
        for phi in phis:
            camera_x = radius * np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180)
            camera_y = radius * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180)
            camera_z = radius * np.cos(phi * np.pi / 180)
            camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
            up = np.array([0, 0, 1], dtype=np.float32)
            lookat = np.array([0, 0, 0], dtype=np.float32)

            pose = build_camera_matrix(
                torch.from_numpy(camera_pos),
                torch.from_numpy(lookat),
                torch.from_numpy(up)
            )
            pose = pose.float().cuda()

            mvp = camera_projmat @ torch.inverse(pose)
            mvp = mvp.cuda().float()

            valid_list, triangle_id_list = rasterize_mesh_depth_peeler(mesh_dict, mvp, glctx, (H, W))

            valid_front, triangle_id_front = valid_list[0], triangle_id_list[0]
            visable_in_mesh = vis_colors[triangle_id_front.reshape(-1)].reshape(H, W)

            alpha = torch.logical_and(visable_in_mesh, valid_front)

            # # visualize the rasterization result:
            # vis_image_alpha = Image.fromarray((alpha.cpu().numpy() * 255).astype(np.uint8))
            # vis_image_valid_front = Image.fromarray((valid_front.cpu().numpy() * 255).astype(np.uint8))
            # vis_image_visable_in_mesh = Image.fromarray((visable_in_mesh.cpu().numpy() * 255).astype(np.uint8))

            # save_dir = "./vis/vis_prune/"
            # os.makedirs(save_dir, exist_ok=True)
            # vis_image_alpha.save(os.path.join(save_dir, f'{int(theta_i):0>3d}_{int(phi):0>3d}_alpha.png'))
            # vis_image_valid_front.save(os.path.join(save_dir, f'{int(theta_i):0>3d}_{int(phi):0>3d}_valid_front.png'))
            # vis_image_visable_in_mesh.save(os.path.join(save_dir, f'{int(theta_i):0>3d}_{int(phi):0>3d}_visable_in_mesh.png'))

            for i in range(len(valid_list)):
                faces_keep[triangle_id_list[i][alpha].reshape(-1).long()] = True

    faces_keep = faces_keep.cpu().numpy()
    print(f'faces_keep: {faces_keep.sum()} / {faces.shape[0]}')

    verts, faces, vert_colors, vis_colors = verts.cpu().numpy(), faces.cpu().numpy(), vert_colors.cpu().numpy(), vis_colors.cpu().numpy()

    verts_map = np.sort(np.unique(faces[faces_keep].reshape(-1)))
    keep = np.zeros((verts.shape[0])).astype(np.bool_)
    keep[verts_map] = True

    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    verts_new = verts[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces_new = faces[keep_faces]

    faces_map = keep_faces

    faces_new[:, 0] = filter_unmapping[faces_new[:, 0]]
    faces_new[:, 1] = filter_unmapping[faces_new[:, 1]]
    faces_new[:, 2] = filter_unmapping[faces_new[:, 2]]

    vert_colors = vert_colors[verts_map]
    vis_colors = vis_colors[faces_map]

    verts_new, faces_new, vert_colors, vis_colors = \
        torch.from_numpy(verts_new).cuda().float(), \
        torch.from_numpy(faces_new).cuda().int(), \
        torch.from_numpy(vert_colors).cuda().float(), \
        torch.from_numpy(vis_colors).cuda() > 0

    return verts_new, faces_new, vert_colors, vis_colors

def refined_obj_bbox(plots_dir, z_floor, delta=0.02, bbox_dist_threshold=0.05):
    '''
    filter floaters and get the refined bbox
    '''

    bbox_root_path = os.path.join(plots_dir, 'bbox')

    mesh_list = os.listdir(plots_dir)
    mesh_list = [x for x in mesh_list if 'surface_' in x]
    epoch_list = [int(x.split('_')[1]) for x in mesh_list]
    epoch = max(epoch_list)
    latest_mesh_list = [x for x in mesh_list if f'surface_{epoch}_' in x]

    for mesh_name in latest_mesh_list:

        if '_0.ply' in mesh_name or '_whole.ply' in mesh_name:      # skip the whole and bg mesh
            continue

        obj_id = (mesh_name.split('.')[0]).split('_')[2]
        bbox_json_path = os.path.join(bbox_root_path, f'bbox_{obj_id}.json')
        if os.path.exists(bbox_json_path):
            os.remove(bbox_json_path)

        mesh_path = os.path.join(plots_dir, mesh_name)
        mesh = trimesh.load(mesh_path)
        max_component = get_max_component_mesh(mesh)
        obj_bbox = get_obj_bbox(max_component, z_floor, delta)

        filtered_mesh = get_filtered_mesh(mesh, obj_bbox, bbox_dist_threshold)
        refined_bbox = get_obj_bbox(filtered_mesh, z_floor, delta)
        with open(bbox_json_path, 'w') as f:
            json.dump(refined_bbox, f)
        print(f'obj {obj_id} refined bbox save to {bbox_json_path}')

        filtered_mesh_path = os.path.join(plots_dir, f'filtered_{mesh_name}')
        filtered_mesh.export(filtered_mesh_path)

def build_camera_matrix(camera_pos, camera_lookat, camera_up):
    camera_z = camera_lookat - camera_pos
    camera_z = camera_z / torch.norm(camera_z)
    camera_x = torch.cross(-camera_up, camera_z)
    camera_x = camera_x / torch.norm(camera_x)
    camera_y = torch.cross(camera_z, camera_x)
    camera_y = camera_y / torch.norm(camera_y)

    camera_matrix = torch.eye(4)
    camera_matrix[:3] = torch.stack([camera_x, camera_y, camera_z, camera_pos], dim=1)
    return camera_matrix


def get_lcc_mesh(verts, faces):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    edges = mesh.edges_sorted.reshape((-1, 2))
    components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
    largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
    verts_map = components[largest_cc].reshape(-1)

    verts_map = np.sort(np.unique(verts_map))
    keep = np.zeros((verts.shape[0])).astype(np.bool_)
    keep[verts_map] = True

    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    verts_lcc = verts[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces_lcc = faces[keep_faces]

    faces_map = keep_faces

    # face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
    faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
    faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]

    return verts_lcc, faces_lcc, verts_map, faces_map

def sample_views_around_object(verts, faces, vert_colors, vis_colors, glctx, resolution, num_views):
    view_dict_list = []
    radius = 1.0
    near = 0.001
    far = 100.0

    visible_threshold = 0.85

    camera_projmat = get_camera_orthogonal_projection_matrix(near, far)
    camera_projmat = torch.from_numpy(camera_projmat).float().cuda()

    mesh_dict = {
        'vertices': F.pad(verts.contiguous(),
                          pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': faces.contiguous(),
        'vert_colors': vert_colors
    }
    H, W = resolution

    num_thetas = 60
    visible_thetas = np.zeros((num_thetas), dtype=np.bool_)
    thetas = np.linspace(0, 360.0, num=num_thetas, endpoint=False)
    phis = np.linspace(-40.0 + 90.0, 40.0 + 90.0, num=9)
    for theta_i, theta in enumerate(thetas):
        visible_theta = False
        for phi in phis:
            camera_x = radius * np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180)
            camera_y = radius * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180)
            camera_z = radius * np.cos(phi * np.pi / 180)
            camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)

            up = np.array([0, 0, 1], dtype=np.float32)
            # up = np.array([
            #     -np.cos(theta * np.pi / 180) * np.cos(phi * np.pi / 180),
            #     -np.sin(theta * np.pi / 180) * np.cos(phi * np.pi / 180),
            #     np.sin(phi * np.pi / 180)
            # ], dtype=np.float32)
            lookat = np.array([0, 0, 0], dtype=np.float32)

            pose = build_camera_matrix(
                torch.from_numpy(camera_pos),
                torch.from_numpy(lookat),
                torch.from_numpy(up)
            )
            pose = pose.float().cuda()

            mvp = camera_projmat @ torch.inverse(pose)
            mvp = mvp.cuda().float()

            colors, valid, triangle_id = rasterize_mesh_vert_colors(mesh_dict, mvp, glctx, resolution)
            visable_in_mesh = vis_colors[triangle_id.reshape(-1)].reshape(H, W)

            visable_triangle_ids_in_view = torch.unique(triangle_id[valid]).reshape(-1)
            visable_triangle_ids_in_view_in_mesh = vis_colors[visable_triangle_ids_in_view]
            visable_ratio = visable_triangle_ids_in_view_in_mesh.float().mean().item()

            alpha = torch.logical_and(visable_in_mesh, valid).float().cpu().numpy().reshape(H, W, 1)

            img = colors.cpu().numpy()
            img = np.concatenate([img, alpha], axis=-1)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

            if visable_ratio > visible_threshold:
                view_dict_list.append({
                    "theta": theta,
                    "phi": phi,
                    "ratio": visable_ratio,
                    "img": img
                })
                visible_theta = True
        visible_thetas[theta_i] = visible_theta

    all_thetas = thetas[visible_thetas]
    print("all_thetas: ", all_thetas)
    selected_thetas = find_longest_contiguous_in_array(visible_thetas)
    selected_thetas = thetas[selected_thetas]
    print("selected_thetas: ", selected_thetas)

    # filter view_dict_list, only maintain those elements that have theta in selected_thetas
    view_dict_list = [view_dict for view_dict in view_dict_list if view_dict['theta'] in selected_thetas]

    # only maintain the largest ratio view for each theta
    view_dict_list = sorted(view_dict_list, key=lambda x: x['ratio'], reverse=True)
    selected_thetas_only = []
    view_dict_list_ = []
    for view_dict in view_dict_list:
        theta = view_dict['theta']
        if theta in selected_thetas_only:
            continue
        selected_thetas_only.append(theta)
        view_dict_list_.append(view_dict)
    view_dict_list = view_dict_list_

    round_degrees = np.zeros((360), dtype=np.bool_)
    bound_angle = 45.0
    for theta in all_thetas:
        left_bound_theta = int(theta - bound_angle)
        right_bound_theta = int(theta + bound_angle)

        if left_bound_theta >= 0 and right_bound_theta < 360:
            round_degrees[left_bound_theta:right_bound_theta] = True
        elif left_bound_theta < 0:
            round_degrees[left_bound_theta:] = True
            round_degrees[:right_bound_theta] = True
        else:
            round_degrees[left_bound_theta:] = True
            round_degrees[:right_bound_theta - 360] = True
    if np.all(round_degrees):
        print("all round degrees are visible")
        view_dict_list = []

    return {
        "view_dict_list": view_dict_list,
        "visible_thetas": all_thetas
    }


def get_faces_normal(verts, faces):

    verts_0 = verts[faces[:, 0]]
    verts_1 = verts[faces[:, 1]]
    verts_2 = verts[faces[:, 2]]

    verts_0_1 = verts_1 - verts_0
    verts_0_2 = verts_2 - verts_0

    normals = np.cross(verts_0_1, verts_0_2)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    return normals

def sample_views_around_object_backface(verts, faces, vert_colors, vis_colors, glctx, resolution, num_views, vis_dir=None):

    faces_front_keep = vis_colors.cpu().numpy() > 0
    verts_np = verts.cpu().numpy()
    faces_np = faces.cpu().numpy()

    verts_map = np.sort(np.unique(faces_np[faces_front_keep].reshape(-1)))
    keep = np.zeros((verts_np.shape[0])).astype(np.bool_)
    keep[verts_map] = True

    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    verts_new = verts_np[keep]
    keep_0 = keep[faces_np[:, 0]]
    keep_1 = keep[faces_np[:, 1]]
    keep_2 = keep[faces_np[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces_new = faces_np[keep_faces]

    faces_map = keep_faces

    faces_new[:, 0] = filter_unmapping[faces_new[:, 0]]
    faces_new[:, 1] = filter_unmapping[faces_new[:, 1]]
    faces_new[:, 2] = filter_unmapping[faces_new[:, 2]]

    front_mesh_dict = {
        'vertices': F.pad(torch.from_numpy(verts_new).cuda().float().contiguous(),
                          pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': torch.from_numpy(faces_new).cuda().int().contiguous(),
    }

    face_normals = get_faces_normal(verts_new, faces_new)
    face_normals = torch.from_numpy(face_normals).cuda().float()

    view_dict_list = []
    radius = 1.0
    near = 0.001
    far = 100.0

    frontface_threshold = 0.995
    visable_threshold = 0.95

    camera_projmat = get_camera_orthogonal_projection_matrix(near, far)
    camera_projmat = torch.from_numpy(camera_projmat).float().cuda()

    mesh_dict = {
        'vertices': F.pad(verts.contiguous(),
                          pad=(0, 1), value=1.0, mode='constant'),
        'pos_idx': faces.contiguous(),
        'vert_colors': vert_colors
    }
    H, W = resolution

    num_thetas = 30
    visible_thetas = np.zeros((num_thetas), dtype=np.bool_)
    thetas = np.linspace(0, 360.0, num=num_thetas, endpoint=False)
    phis = np.linspace(-20.0 + 90.0, 20.0 + 90.0, num=9)

    vis_dir = "./vis" if vis_dir is None else vis_dir
    vis_frontface_dir = f"{vis_dir}/backface_frontface/"
    os.makedirs(vis_frontface_dir, exist_ok=True)
    os.makedirs(f"{vis_dir}/visible/", exist_ok=True)

    for theta_i, theta in enumerate(thetas):
        visible_theta = False
        for phi in phis:
            camera_x = radius * np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180)
            camera_y = radius * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180)
            camera_z = radius * np.cos(phi * np.pi / 180)
            camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)

            up = np.array([0, 0, 1], dtype=np.float32)
            # up = np.array([
            #     -np.cos(theta * np.pi / 180) * np.cos(phi * np.pi / 180),
            #     -np.sin(theta * np.pi / 180) * np.cos(phi * np.pi / 180),
            #     np.sin(phi * np.pi / 180)
            # ], dtype=np.float32)
            lookat = np.array([0, 0, 0], dtype=np.float32)

            pose = build_camera_matrix(
                torch.from_numpy(camera_pos),
                torch.from_numpy(lookat),
                torch.from_numpy(up)
            )
            pose = pose.float().cuda()

            mvp = camera_projmat @ torch.inverse(pose)
            mvp = mvp.cuda().float()

            colors, valid, triangle_id = rasterize_mesh_vert_colors(mesh_dict, mvp, glctx, resolution)
            visable_in_mesh = vis_colors[triangle_id.reshape(-1)].reshape(H, W)

            visable_triangle_ids_in_view = torch.unique(triangle_id[valid]).reshape(-1)
            visable_triangle_ids_in_view_in_mesh = vis_colors[visable_triangle_ids_in_view]
            visable_ratio = visable_triangle_ids_in_view_in_mesh.float().mean().item()

            valid_front, triangle_id_front, _ = rasterize_mesh(front_mesh_dict, mvp, glctx, resolution)
            cam_normals = get_cam_normal_from_rast(valid_front, triangle_id_front, face_normals, pose)
            backface_pixels = cam_normals[..., 2] > 0
            backface_pixels = torch.logical_and(valid_front, backface_pixels)
            frontface_pixels = torch.logical_and(valid_front, torch.logical_not(backface_pixels))

            # visualize front and back pixels with blue and red
            vis_front = np.zeros((H, W, 3))
            vis_front[backface_pixels.cpu().numpy()] = np.array([1, 0, 0])
            vis_front[frontface_pixels.cpu().numpy()] = np.array([0, 0, 1])
            Image.fromarray((vis_front * 255).astype(np.uint8)).save(f'./vis/backface_frontface/{int(theta):0>3d}_{int(phi):0>3d}.png')

            vis_2 = np.zeros((H, W, 3))
            vis_2[(valid > 0).cpu().numpy()] = np.array([1, 0, 0])
            vis_2[(torch.logical_and(visable_in_mesh, valid) > 0).cpu().numpy()] = np.array([0, 0, 1])
            Image.fromarray((vis_2 * 255).astype(np.uint8)).save(f'./vis/visible/{int(theta):0>3d}_{int(phi):0>3d}.png')

            backface_idxs = torch.unique(triangle_id_front[backface_pixels]).reshape(-1)
            frontface_idxs = torch.unique(triangle_id_front[frontface_pixels]).reshape(-1)
            frontface_ratio = frontface_idxs.shape[0] / (frontface_idxs.shape[0] + backface_idxs.shape[0])

            alpha = torch.logical_and(visable_in_mesh, valid).float().cpu().numpy().reshape(H, W, 1)

            img = colors.cpu().numpy()
            img = np.concatenate([img, alpha], axis=-1)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

            # if frontface_ratio > frontface_threshold:
            if frontface_ratio > frontface_threshold or visable_ratio > visable_threshold:
                view_dict_list.append({
                    "theta": theta,
                    "phi": phi,
                    "ratio": visable_ratio,
                    "img": img
                })
                visible_theta = True
        visible_thetas[theta_i] = visible_theta

    all_thetas = thetas[visible_thetas]
    print("all_thetas: ", all_thetas)
    selected_thetas = find_longest_contiguous_in_array(visible_thetas)
    selected_thetas = thetas[selected_thetas]
    print("selected_thetas: ", selected_thetas)

    # filter view_dict_list, only maintain those elements that have theta in selected_thetas
    view_dict_list = [view_dict for view_dict in view_dict_list if view_dict['theta'] in selected_thetas]

    # only maintain the largest ratio view for each theta
    view_dict_list = sorted(view_dict_list, key=lambda x: x['ratio'], reverse=True)
    selected_thetas_only = []
    view_dict_list_ = []
    for view_dict in view_dict_list:
        theta = view_dict['theta']
        if theta in selected_thetas_only:
            continue
        selected_thetas_only.append(theta)
        view_dict_list_.append(view_dict)
    view_dict_list = view_dict_list_

    round_degrees = np.zeros((360), dtype=np.bool_)
    bound_angle = 45.0
    for theta in all_thetas:
        left_bound_theta = int(theta - bound_angle)
        right_bound_theta = int(theta + bound_angle)

        if left_bound_theta >= 0 and right_bound_theta < 360:
            round_degrees[left_bound_theta:right_bound_theta] = True
        elif left_bound_theta < 0:
            round_degrees[left_bound_theta:] = True
            round_degrees[:right_bound_theta] = True
        else:
            round_degrees[left_bound_theta:] = True
            round_degrees[:right_bound_theta - 360] = True
    if np.all(round_degrees):
        print("all round degrees are visible")
        view_dict_list = []


    return {
        "view_dict_list": view_dict_list,
        "visible_thetas": all_thetas
    }

def sample_views_around_object_naive(num_thetas, num_phis):
    view_dict_list = []
    radius = 1.0

    thetas = np.linspace(0, 360.0, num=num_thetas, endpoint=False)
    phis = np.linspace(-30.0 + 90.0, 90.0, num=num_phis)

    for theta in thetas:
        for phi in phis:
            view_dict_list.append({
                "theta": theta,
                "phi": phi,
            })

    return view_dict_list





def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_time():
    torch.cuda.synchronize()
    return time.time()

trans_topil = transforms.ToPILImage()


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

def build_camera_matrix_from_angles_and_locs(theta, phi, scale, shift, radius=1.0):
    camera_x = radius * np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180)
    camera_y = radius * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180)
    camera_z = radius * np.cos(phi * np.pi / 180)
    camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
    camera_pos = apply_inv_scale_shift(camera_pos, scale, shift)

    up = np.array([0, 0, 1], dtype=np.float32)
    lookat = np.array([0, 0, 0], dtype=np.float32)
    lookat = apply_inv_scale_shift(lookat, scale, shift)

    pose = build_camera_matrix(
        torch.from_numpy(camera_pos).float(),
        torch.from_numpy(lookat).float(),
        torch.from_numpy(up).float()
    )
    pose = pose.float()

    return pose

def build_camera_matrix_from_angles_and_locs_diff(theta, phi, scale, shift, shift_2, radius=1.0):
    camera_x = radius * np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180)
    camera_y = radius * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180)
    camera_z = radius * np.cos(phi * np.pi / 180)
    camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
    camera_pos = apply_inv_scale_shift(camera_pos, scale, shift)

    up = np.array([0, 0, 1], dtype=np.float32)
    lookat = np.array([0, 0, 0], dtype=np.float32)
    lookat = apply_inv_scale_shift(lookat, 1.0, shift_2)

    pose = build_camera_matrix(
        torch.from_numpy(camera_pos).float(),
        torch.from_numpy(lookat).float(),
        torch.from_numpy(up).float()
    )
    pose = pose.float()

    return pose

def find_optimal_rotation(n_initial, n_final):
    """
    Find the optimal rotation matrix R that minimizes ||R @ n_i - n_f||

    Parameters:
    -----------
    n_initial : numpy.ndarray
        Initial normals matrix with shape (k, 3)
    n_final : numpy.ndarray
        Final normals matrix with shape (k, 3)

    Returns:
    --------
    R : numpy.ndarray
        Optimal rotation matrix with shape (3, 3)
    """
    # Compute the cross-covariance matrix
    # Transpose n_initial and n_final to get (3, k) shape for matrix multiplication
    H = n_initial.T @ n_final

    # Compute the Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)

    # Construct the rotation matrix
    # The key step is to ensure the rotation preserves right-handedness
    R = Vt.T @ U.T

    # Ensure the rotation matrix is in SO(3)
    # (special orthogonal group - rotation matrices with det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def find_diff_color(colors, epsilon=1e-3, num_candidates=1000):
    """
    Find a color that minimizes the count of similar colors from the input array.

    Parameters:
    colors (numpy.ndarray): Array of colors with shape (N, 3) in the range [0, 1].
    epsilon (float): Threshold for considering two colors as similar.
    num_candidates (int): Number of candidate colors to consider.

    Returns:
    numpy.ndarray: A color with shape (3,) that minimizes the number of similar colors.
    """
    # Calculate the mean color and its complement
    mean_color = np.mean(colors, axis=0)
    complement_mean = 1 - mean_color

    # Generate random candidate colors and include the complement of the mean
    candidates = np.random.rand(num_candidates - 1, 3)
    candidates = np.vstack([candidates, complement_mean])

    # For each candidate, count how many colors in 'colors' have a mean absolute difference less than epsilon
    min_count = float('inf')
    best_c = None

    for c in candidates:
        diffs = np.abs(colors - c)
        count = np.count_nonzero(np.all(diffs < epsilon, axis=-1))

        if count < min_count:
            min_count = count
            best_c = c
        elif count == min_count:
            # If two candidates have the same count, choose the one with the larger average distance
            avg_dist_c = np.mean(diffs)
            avg_dist_best = np.mean(np.mean(np.abs(colors - best_c), axis=1))
            if avg_dist_c > avg_dist_best:
                best_c = c

    return best_c.reshape(3)


def align_normal_pred_lama_omnidata(normal_pred, normal_gt, mask):
    """
    Align predicted normals with ground truth normals using LAMA

    Parameters:
    -----------
    normal_pred : numpy.ndarray
        Predicted normals with shape (h, w, 3)
    normal_gt : numpy.ndarray
        Ground truth normals with shape (h, w, 3)
    mask : numpy.ndarray
        Mask indicating valid pixels with shape (h, w)

    Returns:
    --------
    normal_pred_aligned : numpy.ndarray
        Aligned predicted normals with shape (h, w, 3)
    """
    # Flatten the normals and mask
    n_initial = normal_pred[mask].reshape(-1, 3)
    n_final = normal_gt[mask].reshape(-1, 3)

    # Find the optimal rotation matrix
    R = find_optimal_rotation(n_initial, n_final)

    # Apply the rotation to the predicted normals
    normal_pred_aligned = (R @ normal_pred.reshape(-1, 3).T).T

    normal_pred_aligned =  normal_pred_aligned.reshape(normal_pred.shape)
    normal_pred_aligned[mask] = normal_gt[mask]

    return normal_pred_aligned

def occlusion_test(mesh_a, mesh_b, max_distance=None, ray_offset=1e-5, occlusion_threshold=0.05):
    """
    Test which faces of mesh_a are occluded by mesh_b using ray casting.

    Parameters:
    -----------
    mesh_a : trimesh.Trimesh
        The mesh whose faces will be tested for occlusion
    mesh_b : trimesh.Trimesh
        The potentially occluding mesh

    Returns:
    --------
    occluded_face_indices : numpy.ndarray
        Indices of faces in mesh_a that are occluded by mesh_b
    """

    # Set a reasonable default for max_distance if not provided
    if max_distance is None:
        triangles = mesh_a.vertices[mesh_a.faces.reshape(-1)].reshape(-1, 3, 3)
        edge_01_len = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=-1)
        edge_12_len = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=-1)
        edge_20_len = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=-1)
        average_edge_len = np.mean((edge_01_len + edge_12_len + edge_20_len) / 3)
        max_distance = 30 * average_edge_len

    # Get face centers and normals of mesh_a
    face_centers = mesh_a.triangles_center
    face_normals = mesh_a.face_normals
    n_faces = len(face_centers)

    num_thetas = 15
    num_phis = 7

    rays_per_face = num_thetas * num_phis

    phis = np.linspace(0.0 * np.pi, np.pi * 0.30, num=num_phis, endpoint=True)
    thetas = np.linspace(0., 2. * np.pi, num=num_thetas, endpoint=False)

    phi, theta = np.meshgrid(phis, thetas, indexing='ij')
    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    n = face_normals.reshape(-1, 3).astype(np.float32)
    t = np.array([1., 0., 0.]).reshape(1, 3).repeat(n.shape[0], axis=0)

    t[n[:, 0] >= 0.99] = np.array([0., 1., 0.])
    u = np.cross(t, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    n = n.reshape(-1, 1, 3).repeat(rays_per_face, axis=1)
    u = u.reshape(-1, 1, 3).repeat(rays_per_face, axis=1)
    v = v.reshape(-1, 1, 3).repeat(rays_per_face, axis=1)

    x = x.reshape(1, -1, 1)
    y = y.reshape(1, -1, 1)
    z = z.reshape(1, -1, 1)

    ray_directions = x * u + y * v + z * n
    ray_directions = ray_directions.reshape(-1, 3)

    ray_origins = face_centers.reshape(-1, 3) + ray_offset * face_normals.reshape(-1, 3)
    ray_origins = ray_origins.reshape(-1, 1, 3).repeat(rays_per_face, axis=1).reshape(-1, 3)

    locations, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_b).intersects_location(ray_origins, ray_directions)

    intersect_origin = ray_origins[index_ray]
    intersect_dist = np.linalg.norm(locations - intersect_origin, ord=2, axis=-1)

    valid_intersect = intersect_dist < max_distance
    occluded_face_indices = index_ray[valid_intersect] // rays_per_face

    # unique indices and also get the counts
    occluded_face_indices, counts = np.unique(occluded_face_indices, return_counts=True)

    return occluded_face_indices[counts > occlusion_threshold * rays_per_face]

def get_fg_mask_rembg(image):
    #     image (h, w, 3) [0, 1]
    image_pil = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
    rembg_session = rembg.new_session()


    image_pil_fg = rembg.remove(image_pil, alpha_matting=True, session=rembg_session)
    fg_mask = np.array(image_pil_fg)[..., 3] / 255. > 0

    return fg_mask

def get_theta_phi(eye, to):
    z = eye - to
    z = z / np.linalg.norm(z)
    phi = np.arccos(z[2]) * 180 / np.pi
    theta = np.arctan2(z[1], z[0]) * 180 / np.pi
    if theta < 0:
        theta += 360
    return theta, phi



def smooth_rgb_image(rgb, mask, gaussian_ksize=(13, 13), gaussian_sigma=25.0,
                     morph_kernel_size=3, morph_iterations=1, knn_neighbors=5):
    """
    Smooth the boundary of a masked RGB image to alleviate jagged effects.

    Args:
        rgb: np.float32 (H, W, 3) range 0-1
        mask: np.bool (H, W)
        gaussian_ksize: tuple (width, height), kernel size for Gaussian blur
        gaussian_sigma: float, sigma parameter for Gaussian blur
        morph_kernel_size: int, size of the kernel for dilation/erosion
        morph_iterations: int, number of iterations for dilation/erosion
        knn_neighbors: int, number of neighbors to consider in KNN

    Returns:
        rgb_smooth: Smoothed RGB image
        mask_smooth: Smoothed mask
    """
    # Convert mask to float for processing
    mask_float = mask.astype(np.float32)

    # Apply Gaussian blur to the mask to smooth the boundaries using cv2
    # Note: ksize must be odd numbers for cv2.GaussianBlur
    gaussian_ksize = (gaussian_ksize[0] if gaussian_ksize[0] % 2 == 1 else gaussian_ksize[0] + 1,
                      gaussian_ksize[1] if gaussian_ksize[1] % 2 == 1 else gaussian_ksize[1] + 1)

    mask_blur = cv2.GaussianBlur(mask_float, gaussian_ksize, gaussian_sigma)
    mask_blur = np.where(mask_blur > mask_float, mask_blur, mask_float)

    # Create a boundary region
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=morph_iterations)
    mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=morph_iterations)
    boundary = (mask_dilated.astype(bool) & ~mask_eroded.astype(bool))

    # Get coordinates of boundary pixels and foreground pixels
    boundary_coords = np.argwhere(boundary)
    foreground_coords = np.argwhere(mask & ~boundary)

    # If there are boundary pixels to process
    if len(boundary_coords) > 0 and len(foreground_coords) > 0:
        # Use KNN to find nearest foreground pixels for each boundary pixel
        knn = NearestNeighbors(n_neighbors=knn_neighbors)
        knn.fit(foreground_coords)
        distances, indices = knn.kneighbors(boundary_coords)

        # Create the smoothed RGB image (start with the original)
        rgb_smooth = rgb.copy()

        # For each boundary pixel, calculate weighted average of nearest foreground pixels
        for i, (y, x) in enumerate(boundary_coords):
            # Get the corresponding foreground pixels
            fg_indices = indices[i]
            fg_pixels = [foreground_coords[idx] for idx in fg_indices]

            # Calculate weights based on distance (closer pixels have higher weights)
            weights = 1.0 / (distances[i] + 1e-6)
            weights = weights / np.sum(weights)

            # Calculate weighted average RGB value
            avg_rgb = np.zeros(3)
            for j, (fy, fx) in enumerate(fg_pixels):
                avg_rgb += rgb[fy, fx] * weights[j]

            # Blend with original value based on blurred mask value
            blend_factor = 1.0 - mask_blur[y, x]
            rgb_smooth[y, x] = rgb[y, x] * mask_blur[y, x] + avg_rgb * blend_factor
            # rgb_smooth[y, x] = avg_rgb
    else:
        rgb_smooth = rgb.copy()

    # Create smoothed mask
    mask_smooth = mask_blur > 0.5

    return rgb_smooth, mask_smooth

def find_longest_continuous_azimuths(view_weights):
    num_azimuth_bins, num_altitude_bins = view_weights.shape

    view_weights_azi = np.max(view_weights, axis=1)
    view_weights_azi = view_weights_azi / view_weights_azi.max()
    highlight_azimuths = np.nonzero(view_weights_azi > 0.75)[0]

    center_azimuths = []
    if highlight_azimuths is not None and len(highlight_azimuths) > 0:
        # Ensure all indices are within range using modulo
        highlight_azimuths = [idx % num_azimuth_bins for idx in highlight_azimuths]

        # Sort the indices for processing
        sorted_azimuths = sorted(highlight_azimuths)

        # Find all continuous sequences (considering cyclic nature)
        sequences = []
        current_seq = [sorted_azimuths[0]]

        for i in range(1, len(sorted_azimuths)):
            if sorted_azimuths[i] == (sorted_azimuths[i - 1] + 1) % num_azimuth_bins:
                current_seq.append(sorted_azimuths[i])
            else:
                sequences.append(current_seq)
                current_seq = [sorted_azimuths[i]]

        # Add the last sequence
        sequences.append(current_seq)

        # Check if the first and last sequences form a cyclic continuous sequence
        if len(sequences) > 1:
            first_seq = sequences[0]
            last_seq = sequences[-1]

            if first_seq[0] == (last_seq[-1] + 1) % num_azimuth_bins:
                # Merge the first and last sequences
                merged_seq = last_seq + first_seq
                sequences = sequences[1:-1] + [merged_seq]

        # Find the longest sequence
        longest_seq = max(sequences, key=len)
        highlight_azimuths = longest_seq

        # Find the three central azimuths
        seq_len = len(longest_seq)
        if seq_len >= 3:
            # Get indices for the three central elements
            middle_idx = seq_len // 2
            center_azimuths = [
                longest_seq[middle_idx]
            ]

            # Add one element to the left and right if available
            if seq_len > 1:
                center_azimuths.append(longest_seq[middle_idx - 1])
            if seq_len > 2:
                # For even length sequences, use the element after middle
                # For odd length sequences, use the element after middle+1
                right_idx = middle_idx + 1 if seq_len % 2 == 1 else middle_idx
                if right_idx < seq_len:
                    center_azimuths.append(longest_seq[right_idx])
        else:
            # If sequence length < 3, use all available elements
            center_azimuths = longest_seq.copy()

    return highlight_azimuths, center_azimuths

def visualize_graph_tree(graph_node_dict, output_file="graph_tree", format="png"):
    """
    Visualize the tree structure of the graph based on the graph_node_dict.

    Parameters:
    graph_node_dict (dict): Dictionary containing node properties
    output_file (str): Name of the output file (without extension)
    format (str): Output format ('png', 'svg', 'pdf', etc.)

    Returns:
    graphviz.Digraph: The generated graph visualization
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Graph Tree Visualization')

    # Set graph attributes for layout
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr(nodesep='0.5')
    dot.attr(ranksep='1.0')

    # Group nodes by layer
    nodes_by_layer = defaultdict(list)
    for node_id, properties in graph_node_dict.items():
        layer = properties['layer']
        nodes_by_layer[layer].append(node_id)

    # Create subgraphs for each layer to align nodes
    for layer, nodes in sorted(nodes_by_layer.items(), reverse=True):
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.attr(label=f'Layer {layer}')

            # Add nodes to this layer
            for node_id in sorted(nodes):
                props = graph_node_dict[node_id]

                # Determine node style based on properties
                node_style = {
                    'shape': 'circle',
                    'style': 'filled',
                    'fontname': 'Arial',
                    'fontsize': '12',
                }

                # Different colors for different node types
                if props['root']:
                    node_style.update({
                        'fillcolor': '#ff9900',
                        'penwidth': '2.0',
                    })
                elif props['leaf']:
                    node_style.update({
                        'fillcolor': '#66ccff',
                        'penwidth': '1.0',
                    })
                else:
                    node_style.update({
                        'fillcolor': '#99cc99',
                        'penwidth': '1.0',
                    })

                # Add the node with its style
                dot.node(str(node_id), str(node_id), **node_style)

    # Add edges (parent -> child relationships)
    for node_id, props in graph_node_dict.items():
        parent = props['parent']
        if parent != -1:  # Skip the root node which has parent -1
            dot.edge(str(parent), str(node_id))

    # Render the graph
    dot.render(output_file, format=format, cleanup=True)

    print(f"Graph visualization saved as {output_file}.{format}")
    return dot



def visualize_view_weights_with_highlighted_azimuths(view_weights, points=None, highlight_azimuths=None, center_azimuths=None, output_path='view_weights_visualization.png',
                           dpi=300, altitude_range=(60, 120), highlight_color='blue', center_highlight_color='red',
                           highlight_linewidth=2, center_linewidth=3):
    """
    Visualize view weights on a spherical projection and save to a PNG file.
    Highlights the longest continuous sequence of azimuth indices, with special highlighting
    for the three most central azimuths in that sequence.

    Parameters:
    view_weights: numpy array of shape (num_azimuth_bins, num_altitude_bins)
    highlight_azimuths: list of azimuth indices to highlight with vertical lines (default: None)
    output_path: path to save the PNG file (default: 'view_weights_visualization.png')
    dpi: resolution of the output image (default: 300)
    altitude_range: range of altitude angles in degrees (default: (60, 120))
    highlight_color: color for the highlight lines (default: 'blue')
    center_highlight_color: color for the central highlight lines (default: 'red')
    highlight_linewidth: width of the highlight lines (default: 2)
    center_linewidth: width of the central highlight lines (default: 3)
    """
    # Get dimensions
    num_azimuth_bins, num_altitude_bins = view_weights.shape

    # Create figure with Mollweide projection (good for spherical visualization)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='mollweide')

    # Create a meshgrid for azimuth and altitude
    # Convert bin indices to radians for Mollweide projection
    phi = np.linspace(-np.pi, np.pi, num_azimuth_bins)
    theta = np.linspace((altitude_range[1] - 90.0) * np.pi / 180.0, (altitude_range[0] - 90.0) * np.pi / 180.0,
                        num_altitude_bins)

    # Create 2D arrays for pcolormesh
    Phi, Theta = np.meshgrid(phi, theta)

    # We need to properly orient the data for display
    # For Mollweide projection, we need the data in the right shape
    weights_display = view_weights

    # Create custom orange-red colormap
    colors = [(1, 1, 1, 0),  # White (transparent) for low values
              (1, 0.6, 0.2, 0.5),  # Light orange for medium values
              (1, 0.3, 0, 1)]  # Dark orange/red for high values
    orange_cmap = LinearSegmentedColormap.from_list('OrangeRed', colors)

    # Plot the data with correct orientation
    # In Mollweide projection, we need to be careful about the orientation
    # We need to transpose and flip appropriately
    mesh = ax.pcolormesh(Phi, Theta, weights_display.T, cmap=orange_cmap, shading='auto')

    # Add grid lines
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')

    # Set azimuth ticks (convert to degrees for display)
    azimuth_ticks = np.array([30 - 180, 120 - 180, 180 - 180, 240 - 180, 330 - 180]) * np.pi / 180
    ax.set_xticks(azimuth_ticks)
    ax.set_xticklabels(['$30°$', '$120°$', '$180°$', '$240°$', '$330°$'])

    # Remove y-axis ticks to match the reference image
    ax.set_yticks([])

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Create a thin horizontal line at the equator (theta=0)
    ax.axhline(y=0, color='black', alpha=0.7, linewidth=1.5)

    # Highlight specified azimuth indices
    if highlight_azimuths is not None:
        # Get the theta range for drawing vertical lines
        theta_min = np.min(Theta)
        theta_max = np.max(Theta)

        # Draw vertical lines for each specified azimuth index
        for az_idx in highlight_azimuths:
            # Get the corresponding azimuth angle in radians
            az_angle = phi[az_idx]

            # Check if this is one of the center azimuths
            is_center = az_idx in center_azimuths

            # Set color and linewidth based on whether this is a center azimuth
            line_color = center_highlight_color if is_center else highlight_color
            line_width = center_linewidth if is_center else highlight_linewidth

            # Draw a vertical line at this azimuth angle
            ax.plot([az_angle, az_angle], [theta_min, theta_max],
                    color=line_color, linewidth=line_width,
                    linestyle='-', alpha=0.8)

            # Add a small marker at the top of each line for better visibility
            marker_style = '^' if is_center else 'v'
            marker_size = 8 if is_center else 6
            ax.plot(az_angle, theta_min, marker=marker_style, color=line_color,
                    markersize=marker_size, alpha=0.8)

    # Plot specific points if provided
    if points is not None and len(points) > 0:
        # Convert points from (azimuth, altitude) in degrees to (phi, theta) in radians
        point_coords = []
        for azimuth, altitude in points:
            # Convert azimuth from 0-360° to -180°-+180° (or -π to π)
            phi_val = (azimuth - 180) * np.pi / 180

            # Convert altitude from 0-180° to theta (-π/2 to π/2)
            # In Mollweide projection, theta is 0 at equator, positive toward north pole, negative toward south pole
            theta_val = (90 - altitude) * np.pi / 180

            point_coords.append((phi_val, theta_val))

        # Plot each point
        phi_vals = [p[0] for p in point_coords]
        theta_vals = [p[1] for p in point_coords]

        ax.scatter(phi_vals, theta_vals, s=40, c='lime', marker='o',
                   edgecolors='black', linewidths=1, zorder=5, alpha=0.9)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Tight layout
    plt.tight_layout()

    # Save figure to file
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)

    return output_path

def occlusion_test_point_to_mesh(points, normals, mesh_b, max_distance=None, ray_offset=1e-5, occlusion_threshold=0.0001):
    """
    Test which faces of mesh_a are occluded by mesh_b using ray casting.

    Parameters:
    -----------
    mesh_a : trimesh.Trimesh
        The mesh whose faces will be tested for occlusion
    mesh_b : trimesh.Trimesh
        The potentially occluding mesh

    Returns:
    --------
    occluded_face_indices : numpy.ndarray
        Indices of faces in mesh_a that are occluded by mesh_b
    """

    # Set a reasonable default for max_distance if not provided
    points = points.reshape(-1, 3)
    normals = normals.reshape(-1, 3)

    if max_distance is None:
        # calculate the average minimum distance among points
        # find the closest point with KDtree KNN first
        kdtree = KDTree(points)
        dists, _ = kdtree.query(points, k=2)
        dists = dists[:, 1]
        average_min_distance = np.mean(dists)
        max_distance = 50 * average_min_distance

    # Get face centers and normals of mesh_a
    face_centers = points
    face_normals = normals

    num_thetas = 30
    num_phis = 50

    rays_per_face = num_thetas * num_phis

    phis = np.linspace(0.0 * np.pi, np.pi * 0.45, num=num_phis, endpoint=True)
    thetas = np.linspace(0., 2. * np.pi, num=num_thetas, endpoint=False)

    phi, theta = np.meshgrid(phis, thetas, indexing='ij')
    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    n = face_normals.reshape(-1, 3).astype(np.float32)
    t = np.array([1., 0., 0.]).reshape(1, 3).repeat(n.shape[0], axis=0)

    t[n[:, 0] >= 0.99] = np.array([0., 1., 0.])
    u = np.cross(t, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    n = n.reshape(-1, 1, 3).repeat(rays_per_face, axis=1)
    u = u.reshape(-1, 1, 3).repeat(rays_per_face, axis=1)
    v = v.reshape(-1, 1, 3).repeat(rays_per_face, axis=1)

    x = x.reshape(1, -1, 1)
    y = y.reshape(1, -1, 1)
    z = z.reshape(1, -1, 1)

    ray_directions = x * u + y * v + z * n
    ray_directions = ray_directions.reshape(-1, 3)

    ray_origins = face_centers.reshape(-1, 3) + ray_offset * face_normals.reshape(-1, 3)
    ray_origins = ray_origins.reshape(-1, 1, 3).repeat(rays_per_face, axis=1).reshape(-1, 3)

    locations, index_ray, _ = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_b).intersects_location(ray_origins, ray_directions)

    intersect_origin = ray_origins[index_ray]
    intersect_dist = np.linalg.norm(locations - intersect_origin, ord=2, axis=-1)

    valid_intersect = intersect_dist < max_distance
    occluded_face_indices = index_ray[valid_intersect] // rays_per_face

    # unique indices and also get the counts
    occluded_face_indices, counts = np.unique(occluded_face_indices, return_counts=True)

    return occluded_face_indices[counts > occlusion_threshold * rays_per_face]


def get_fg_occulusion_mask(depth, normal, pose, scale, mask, sub_meshes):
    H, W = depth.shape[:2]
    near = 0.001
    ray_o, ray_d = get_camera_orthogonal_rays(H, W, near, torch.from_numpy(pose), scale)
    mask = mask.reshape(-1)
    depth = depth.reshape(-1)
    ray_o = ray_o.reshape(-1, 3).cpu().numpy()
    ray_d = ray_d.reshape(-1, 3).cpu().numpy()

    ray_o = ray_o[mask]
    ray_d = ray_d[mask]
    depth = depth[mask]
    normals = normal.reshape(-1, 3)[mask]

    normals = (pose[:3, :3] @ normals.T).T

    points = ray_o + ray_d * depth.reshape(-1, 1)

    occluded_idxs = occlusion_test_point_to_mesh(points, normals, trimesh.util.concatenate(sub_meshes))
    print("occluded_idxs: ", occluded_idxs.shape)
    occluded_mask = np.zeros(H*W, dtype=np.bool_)
    occluded_mask[np.arange(H*W)[mask][occluded_idxs]] = True

    return occluded_mask.reshape(H, W)

def generate_traverse_seq(graph_node_dict, num_objs):

    traverse_list = []
    root = 0
    for obj_i in range(1, num_objs):
        # _obj_i_cur = obj_i
        # traverse_list.append(_obj_i_cur)
        # while _obj_i_cur != root:
        #     _obj_i_cur = graph_node_dict[_obj_i_cur]['parent']
        #     if _obj_i_cur != root:
        #         traverse_list.append(_obj_i_cur)
        _obj_i_cur = obj_i
        layer = 1
        while _obj_i_cur != root:
            for _ in range(layer):
                traverse_list.append(_obj_i_cur)
            _obj_i_cur = graph_node_dict[_obj_i_cur]['parent']
            layer += 1
    return traverse_list

def make_sphere(level:int=2,radius=1.,device='cuda'):
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=0.2, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return vertices,faces



def _translation(tx, ty, tz, device):
    translation_matrix = torch.eye(4, device=device)
    translation_matrix[0, 3] = tx
    translation_matrix[1, 3] = ty
    translation_matrix[2, 3] = tz
    return translation_matrix

def _projection(r, device):
    projection_matrix = torch.eye(4, device=device)
    projection_matrix[3, 2] = -r
    return projection_matrix



def _orthographic(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    o = torch.zeros([4,4],device=device)
    o[0,0] = 2/(r-l)
    o[0,3] = -(r+l)/(r-l) # 0
    o[1,1] = 2/(t-b) * (-1 if flip_y else 1)
    o[1,3] = -(t+b)/(t-b) # 0
    o[2,2] = -2/(f-n)
    o[2,3] = -(f+n)/(f-n)
    o[3,3] = 1
    return o # 4,4

def make_star_cameras_fixed_angles(distance: float = 10., r: float = None, image_size=[512, 512], device='cuda', angles=None):
    if r is None:
        r = 1 / distance

    # Define the fixed angles in degrees
    # if not angles:
    angles = [0, -45, -90, 180, 90, 45]
    # angles = [0, -90, 180, 90]
    angles_rad = [angle * torch.pi / 180.0 for angle in angles]
    C = len(angles)

    mv = torch.eye(4, device=device).repeat(C, 1, 1)

    for i, angle in enumerate(angles_rad):
        rot = torch.eye(3, device=device)
        angle = torch.tensor(angle)
        rot[2, 2] = torch.cos(angle)
        rot[2, 0] = -torch.sin(angle)
        rot[0, 2] = torch.sin(angle)
        rot[0, 0] = torch.cos(angle)

        mv[i, :3, :3] = rot

    # Apply translation
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r, device)

def make_wonder3D_cameras(distance: float=10. , r: float=1., image_size = [512, 512], device = 'cuda', cam_type='ortho', angles=None):
    mv, _ = make_star_cameras_fixed_angles(angles=angles)
    r = 1
    print('making orthogonal cameras')
    return mv, _orthographic(r, device)


def get_axis_rotate_matrix(vec, angle):
    """
    Compute the rotation matrix for rotating around a unit vector by a given angle.

    Parameters:
    vec (numpy.ndarray): A 3D unit vector defining the rotation axis
    angle (float): The rotation angle in radians

    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    # Ensure the vector is a unit vector
    vec = np.array(vec, dtype=np.float64)
    vec = vec / np.linalg.norm(vec)

    # Extract components
    x, y, z = vec

    # Compute sine and cosine of the angle
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    # Construct the rotation matrix using Rodrigues' rotation formula
    rotation_matrix = np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c]
    ])

    return rotation_matrix


def w_n2c_n(imgs, mv):
    # imgs: (B,H,W,C)
    # mv: (B,4,4), world to camera

    B,H,W,C = imgs.shape
    img_masks = imgs[..., 3:]
    normals = imgs[...,:3].reshape(B, -1, 3) * 2 - 1
    imgs_cam_coord = (normals @ mv[:,:3,:3].transpose(2, 1))[...,:3]
    imgs_cam_coord /= torch.norm(imgs_cam_coord, dim=-1, keepdim=True)
    imgs_cam_coord = imgs_cam_coord.reshape(B,H,W,3) * 0.5 + 0.5
    if img_masks.shape[0] == 0:
        img_masks_nag = (imgs[...,0] < 0.3) & (imgs[...,1] < 0.3) & (imgs[...,2] < 0.3)
        img_masks = (~img_masks_nag[...,None]).to(imgs)

    imgs_cam_coord = torch.concat([imgs_cam_coord*img_masks, img_masks], dim=-1)
    return imgs_cam_coord


def grid_sample_3d_with_channels(tensor, p):
    """
    Perform trilinear interpolation on a 3D tensor with channels.

    Args:
        tensor: A 4D tensor of shape (H, W, D, C)
        p: A 2D tensor of shape (N, 3) containing N 3D coordinates
           p[:, 0] is in [0, H-1], p[:, 1] is in [0, W-1], p[:, 2] is in [0, D-1]

    Returns:
        A tensor of shape (N, C) containing the interpolated values
    """
    H, W, D, C = tensor.shape
    N = p.shape[0]

    # Get integer indices and fractional parts
    h = p[:, 0]
    w = p[:, 1]
    d = p[:, 2]

    h0 = torch.floor(h).long()
    w0 = torch.floor(w).long()
    d0 = torch.floor(d).long()

    h1 = h0 + 1
    w1 = w0 + 1
    d1 = d0 + 1

    # Ensure indices are within bounds
    h0 = torch.clamp(h0, 0, H - 1)
    w0 = torch.clamp(w0, 0, W - 1)
    d0 = torch.clamp(d0, 0, D - 1)
    h1 = torch.clamp(h1, 0, H - 1)
    w1 = torch.clamp(w1, 0, W - 1)
    d1 = torch.clamp(d1, 0, D - 1)

    # Get fractional part
    h_frac = h - h0.float()
    w_frac = w - w0.float()
    d_frac = d - d0.float()

    # Expand dimensions for broadcasting
    h_frac = h_frac.view(N, 1)
    w_frac = w_frac.view(N, 1)
    d_frac = d_frac.view(N, 1)

    # Flatten the tensor for indexing
    tensor_flat = tensor.reshape(-1, C)
    stride_h = W * D
    stride_w = D
    stride_d = 1

    # Compute linear indices for the 8 neighboring points
    idx000 = h0 * stride_h + w0 * stride_w + d0 * stride_d
    idx001 = h0 * stride_h + w0 * stride_w + d1 * stride_d
    idx010 = h0 * stride_h + w1 * stride_w + d0 * stride_d
    idx011 = h0 * stride_h + w1 * stride_w + d1 * stride_d
    idx100 = h1 * stride_h + w0 * stride_w + d0 * stride_d
    idx101 = h1 * stride_h + w0 * stride_w + d1 * stride_d
    idx110 = h1 * stride_h + w1 * stride_w + d0 * stride_d
    idx111 = h1 * stride_h + w1 * stride_w + d1 * stride_d

    # Gather values
    val000 = tensor_flat[idx000]
    val001 = tensor_flat[idx001]
    val010 = tensor_flat[idx010]
    val011 = tensor_flat[idx011]
    val100 = tensor_flat[idx100]
    val101 = tensor_flat[idx101]
    val110 = tensor_flat[idx110]
    val111 = tensor_flat[idx111]

    # Perform trilinear interpolation
    c00 = val000 * (1 - h_frac) + val100 * h_frac
    c01 = val001 * (1 - h_frac) + val101 * h_frac
    c10 = val010 * (1 - h_frac) + val110 * h_frac
    c11 = val011 * (1 - h_frac) + val111 * h_frac

    c0 = c00 * (1 - w_frac) + c10 * w_frac
    c1 = c01 * (1 - w_frac) + c11 * w_frac

    output = c0 * (1 - d_frac) + c1 * d_frac

    return output


def grid_sample_3d(tensor, points):

    return grid_sample_3d_with_channels(tensor.unsqueeze(-1), points).view(-1)

def coarse_recon(view_list, scale, center, R_from_w3d, debug_dir=None):
    vertices_init, faces_init = make_sphere(radius=3)
    # vertices_init = vertices_init * scale + center.cuda()
    # return trimesh.Trimesh(vertices=vertices_init.cpu().numpy(), faces=faces_init.cpu().numpy())

    mv_normal_list = []
    proj_normal_list = []
    gt_normals_world = []
    RGBs = []

    for view in view_list:
        normal = view['normal']
        rgb = view['rgb']
        mask = view['mask']
        pose = view['pose']
        scale = view['scale']
        mv = view['mv']
        proj = view['proj']

        pose = pose.cuda().clone()

        # proj = get_camera_orthogonal_projection_matrix(0.001, 2.0, scale)
        # proj = torch.from_numpy(proj).cuda().float()
        # proj_normal_list.append(proj)
        #
        # mv_normal_list.append(torch.inverse(pose).float().cuda())

        mv_normal_list.append(mv.float().cuda())
        proj_normal_list.append(proj.float().cuda())

        H, W = normal.shape[:2]
        normal = torch.from_numpy(normal).float().cuda()
        mask = torch.from_numpy(mask).float().cuda()

        normal = normal.clone()
        normal[..., 1:3] *= -1
        normal = normal

        gt_normal_world = (torch.inverse(mv)[:3, :3] @ normal.reshape(-1, 3).T).T.reshape(H, W, 3)
        gt_normal_world = torch.nn.functional.normalize(gt_normal_world, p=2, dim=-1)
        gt_normal_world = gt_normal_world * 0.5 + 0.5
        gt_normal_world = torch.cat([gt_normal_world, mask.reshape(H, W, 1)], dim=-1)
        gt_normals_world.append(gt_normal_world)

        RGBs.append(
            torch.cat([torch.from_numpy(rgb).float().cuda(), mask.reshape(H, W, 1)], dim=-1)
        )

    mv_normal = torch.stack(mv_normal_list, dim=0)
    proj_normal = torch.stack(proj_normal_list, dim=0)
    gt_normals_world = torch.stack(gt_normals_world, dim=0)
    gt_RGBs = torch.stack(RGBs, dim=0)

    renderer = Renderer(mv_normal, proj_normal, [H, W])

    # normals = w_n2c_n(gt_normals_world, mv_normal)
    # vertices_init, faces_init = CoarseRecon(front_normal=normals[0], back_normal=normals[3], side_normal=normals[2],
    #                                         output_path=debug_dir, is_persp=False)
    #
    # trimesh.exchange.export.export_mesh(
    #     trimesh.Trimesh(vertices=vertices_init.cpu().numpy(), faces=vertices_init.cpu().numpy()),
    #     os.path.join(debug_dir, 'coarse_mesh_1.ply'),
    # )

    vertices, faces = do_optimize(vertices_init, faces_init, gt_normals_world, renderer, None, edge_len_lims=(0.01, 0.05), remeshing_steps=200, debug_dir=debug_dir)
    # print("vertices: ", vertices.shape, vertices)
    # print("faces: ", faces.shape, faces)

    if torch.any(torch.isnan(vertices)):
        return None, None

    # print("finish do optimize")
    vertices, faces = geo_aware_mesh_refine(vertices, faces, gt_normals_world, renderer, mv_normal,
                                            proj_normal, 100, start_edge_len=0.01,
                                            end_edge_len=0.008, b_persp=False, update_normal_interval=10,
                                            update_warmup=5)
    # print("finish geo aware mesh refine")
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=1,
                               apply_sub_divide=True, sub_divide_threshold=0.25).to("cuda")

    vertices = meshes._verts_list[0]
    faces = meshes._faces_list[0]
    # print("vertices: ", vertices.shape)
    # print("faces: ", faces.shape)

    RGB_view_weights = np.array([2.0, 0.05, 0.2, 1.0, 0.2, 0.05])
    RGB_refine_index = np.arange(len(mv_normal))

    textured_mesh = opt_warpper(vertices, faces, gt_RGBs, mv_normal, proj_normal, weights=RGB_view_weights, refine_index=RGB_refine_index, render_size=gt_RGBs.shape[-2], b_perspective=False, visualize=True, do_refine_uv=False)
    vt = textured_mesh.mesh.v_tex,
    ft = textured_mesh.mesh.t_tex_idx,
    texture_map = textured_mesh.map_Kd[..., :3]
    vt = vt[0]
    ft = ft[0]
    vertices = textured_mesh.mesh.v_pos

    vertices = vertices @ R_from_w3d.T
    vertices = vertices * scale + center.cuda()

    faces = textured_mesh.mesh.t_pos_idx
    # vt[:, 1] = 1 - vt[:, 1]
    # save_obj(
    #     os.path.join(debug_dir, 'textured_mesh.obj'),
    #     vertices.reshape(-1, 3).cpu(),
    #     faces.reshape(-1, 3).cpu(),
    #     verts_uvs = vt.reshape(-1, 2).cpu(),
    #     faces_uvs = ft.reshape(-1, 3).cpu(),
    #     texture_map = texture_map.cpu()
    # )
    texture_dict = {
        'vt': vt.cpu().numpy(),
        'ft': ft.cpu().numpy(),
        'texture_map': texture_map.cpu().numpy()
    }

    return trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy()), texture_dict


def clean_mesh_floaters_adjust(mesh, min_ratio=0.02):
    verts, faces = mesh.vertices, mesh.faces
    has_color = False
    try:
        vert_colors = mesh.visual.vertex_colors[..., :3].reshape(-1, 3) / 255.
        has_color = True
    except:
        pass
    edges = mesh.edges_sorted.reshape((-1, 2))
    components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
    n_components = len(components)

    components = [comp for comp in components if len(comp) > min_ratio * verts.shape[0]]
    if len(components) == n_components or len(components) == 0:
        return mesh
    verts_map = np.concatenate(components, axis=0).reshape(-1)

    verts_map = np.sort(np.unique(verts_map))
    keep = np.zeros((verts.shape[0])).astype(np.bool_)
    keep[verts_map] = True

    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    verts_lcc = verts[keep]

    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces_lcc = faces[keep_faces]

    faces_map = keep_faces

    # face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
    faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
    faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]

    if has_color:
        color_lcc = vert_colors[verts_map]
        return trimesh.Trimesh(verts_lcc, faces_lcc, vertex_colors=color_lcc, process=False)

    return trimesh.Trimesh(verts_lcc, faces_lcc, process=False)


def marching_cubes_from_sdf(model, obj_id=None, resolution=512, grid_boundary=(-1, 1), level=0.00):
    assert obj_id is not None
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    with torch.no_grad():
        z_all = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z_all.append(model.get_sdf_raw(pnts.cuda())[:, obj_id].detach().cpu().numpy().reshape(-1))
        sdf_z = np.concatenate(z_all, axis=0)

        if (not (np.min(sdf_z) > level or np.max(sdf_z) < level)):
            sdf_z = sdf_z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=sdf_z.reshape(x.shape[0], y.shape[0], z.shape[0]),
                level=level,
                spacing=(x[2] - x[1],
                         y[2] - y[1],
                         z[2] - z[1]))

            verts = verts + np.array([x[0], y[0], z[0]])

            meshexport = trimesh.Trimesh(verts, faces, process=False)

            return meshexport

def simplify_mesh(mesh_trimesh, target_faces):
    pml_mesh = pml.Mesh(mesh_trimesh.vertices, mesh_trimesh.faces)
    ms = pml.MeshSet()
    ms.add_mesh(pml_mesh, 'mesh')

    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_faces)

    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    return trimesh.Trimesh(verts, faces, process=False)



def detect_collision(base_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], test_mesh: Tuple[np.ndarray, np.ndarray]):
    """
    Detect collisions between a test mesh and a series of base meshes.
    Uses edge-based ray casting to detect intersections.

    Parameters:
    -----------
    base_meshes : List of tuples, each containing (vertices, faces)
        List of base meshes to check against.
        vertices: (n, 3) float32 - Vertex coordinates
        faces: (m, 3) int32 - Face indices
        face_normals: (m, 3) float32 - Face normals (optional, can be None)

    test_mesh : Tuple of (vertices, faces)
        The mesh to test for collisions.
        vertices: (n, 3) float32 - Vertex coordinates
        faces: (m, 3) int32 - Face indices

    Returns:
    --------
    contact_points : np.ndarray
        (k, 3) array of contact point coordinates
    contact_mesh_id : np.ndarray
        (k,) array of indices indicating which base mesh had the contact
    contact_face_id : np.ndarray
        (k,) array of face indices in the base mesh
    """
    # Convert test mesh to trimesh object
    test_vertices, test_faces = test_mesh
    test_trimesh = trimesh.Trimesh(vertices=test_vertices, faces=test_faces)

    # Extract edges from test mesh
    edges = test_trimesh.edges_unique

    # Get edge vertices
    edge_points = test_vertices[edges]

    # Create ray origins and directions from edges
    ray_origins = edge_points[:, 0]
    ray_directions = edge_points[:, 1] - edge_points[:, 0]

    # Normalize ray directions
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, np.newaxis]

    # Lists to store contact information
    all_contact_points = []
    all_contact_mesh_ids = []
    all_contact_face_ids = []
    all_contact_face_normals = []

    # Check collision with each base mesh
    for mesh_id, (base_vertices, base_faces, base_face_normals) in enumerate(base_meshes):
        # Create trimesh object for the base mesh
        base_trimesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)

        # Use ray_mesh intersection to find contacts
        # locations, index_ray, index_tri = base_trimesh.ray.intersects_location(
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions
        # )

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(base_trimesh).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) > 0:
            # Calculate distances from ray origins to intersection points
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

            # Only consider intersections that fall within the edge length
            valid_indices = distances <= ray_lengths[index_ray]

            if np.any(valid_indices):
                contact_points = locations[valid_indices]
                contact_face_ids = index_tri[valid_indices]
                contact_mesh_ids = np.full(len(contact_points), mesh_id)
                contact_normals = base_face_normals[contact_face_ids]

                all_contact_points.append(contact_points)
                all_contact_mesh_ids.append(contact_mesh_ids)
                all_contact_face_ids.append(contact_face_ids)
                all_contact_face_normals.append(contact_normals)

    # Combine results from all base meshes
    if all_contact_points:
        contact_points = np.vstack(all_contact_points)
        contact_mesh_id = np.concatenate(all_contact_mesh_ids)
        contact_face_id = np.concatenate(all_contact_face_ids)
        contact_face_normals = np.vstack(all_contact_face_normals)
    else:
        # Return empty arrays if no contacts found
        contact_points = np.empty((0, 3), dtype=np.float32)
        contact_mesh_id = np.empty(0, dtype=np.int32)
        contact_face_id = np.empty(0, dtype=np.int32)
        contact_face_normals = np.empty((0, 3), dtype=np.float32)

    return contact_points, contact_mesh_id, contact_face_id, contact_face_normals

def pair_mesh_collision(mesh1, mesh2):
    """
    test whether mesh1 has collision with mesh2
    if so, return the contact points, face ids and face normals of mesh1
    """

    mesh1_vertices, mesh1_faces = mesh1.vertices, mesh1.faces
    mesh2_vertices, mesh2_faces = mesh2.vertices, mesh2.faces

    mesh1_face_normals = mesh1.face_normals

    contact_points, contact_mesh_id, contact_face_id, contact_face_normals = detect_collision(
        [(mesh1_vertices, mesh1_faces, mesh1_face_normals)], 
        (mesh2_vertices, mesh2_faces)
    )

    return {
        'collision': contact_points.shape[0] > 0,
        'contact_points': contact_points,
        'contact_face_normals': contact_face_normals
    }

def falldown_collision(base_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], test_mesh: Tuple[np.ndarray, np.ndarray]):
    # Convert test mesh to trimesh object
    test_vertices, test_faces = test_mesh
    test_trimesh = trimesh.Trimesh(vertices=test_vertices, faces=test_faces)
    

    # generate ray downs
    ray_origins = test_vertices
    ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(test_vertices.shape[0], axis=0)

    locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(test_trimesh).intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )
    
    average_edge_len = float(np.mean(np.linalg.norm(test_vertices[test_faces[:, 0]] - test_vertices[test_faces[:, 1]], axis=1)))

    index_ray_no_hit = np.ones(ray_origins.shape[0], dtype=np.bool_)
    index_ray_no_hit[index_ray] = False
    ray_origins = ray_origins[index_ray_no_hit]
    ray_directions = ray_directions[index_ray_no_hit]

        # Lists to store contact information
    all_contact_points = []
    all_contact_mesh_ids = []
    all_contact_face_ids = []
    all_contact_face_normals = []

    min_distance_all = +np.inf

    # Check collision with each base mesh
    for mesh_id, (base_vertices, base_faces, base_face_normals) in enumerate(base_meshes):
        # Create trimesh object for the base mesh
        base_trimesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)

        # Use ray_mesh intersection to find contacts
        # locations, index_ray, index_tri = base_trimesh.ray.intersects_location(
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions
        # )

        locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(base_trimesh).intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) > 0:
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
            min_distance = np.min(distances)

            if min_distance < min_distance_all:
                min_distance_all = min_distance

                contact_points = locations
                contact_face_ids = index_tri
                contact_mesh_ids = np.full(len(contact_points), mesh_id)
                contact_normals = base_face_normals[contact_face_ids]

                all_contact_points = [contact_points]
                all_contact_mesh_ids = [contact_mesh_ids]
                all_contact_face_ids = [contact_face_ids]
                all_contact_face_normals = [contact_normals]

    # Combine results from all base meshes
    if all_contact_points:
        contact_points = np.vstack(all_contact_points)
        contact_mesh_id = np.concatenate(all_contact_mesh_ids)
        contact_face_id = np.concatenate(all_contact_face_ids)
        contact_face_normals = np.vstack(all_contact_face_normals)
    else:
        # Return empty arrays if no contacts found
        contact_points = np.empty((0, 3), dtype=np.float32)
        contact_mesh_id = np.empty(0, dtype=np.int32)
        contact_face_id = np.empty(0, dtype=np.int32)
        contact_face_normals = np.empty((0, 3), dtype=np.float32)

    return contact_points, contact_mesh_id, contact_face_id, contact_face_normals



def falldown_collision_merged(base_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], test_mesh: Tuple[np.ndarray, np.ndarray]):
    # Convert test mesh to trimesh object
    test_vertices, test_faces = test_mesh
    test_trimesh = trimesh.Trimesh(vertices=test_vertices, faces=test_faces)
    

    # generate ray downs
    ray_origins = test_vertices
    ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(test_vertices.shape[0], axis=0)

    locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(test_trimesh).intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
    average_edge_len = float(np.mean(np.linalg.norm(test_vertices[test_faces[:, 0]] - test_vertices[test_faces[:, 1]], axis=1)))

    index_ray_no_hit = np.ones(ray_origins.shape[0], dtype=np.bool_)
    index_ray_no_hit[index_ray] = False
    index_ray_no_hit[index_ray[distances > 4.0 * average_edge_len]] = False
    ray_origins = ray_origins[index_ray_no_hit]
    ray_directions = ray_directions[index_ray_no_hit]

    # Lists to store contact information
    all_contact_points = []
    all_contact_mesh_ids = []
    all_contact_face_ids = []
    all_contact_face_normals = []

    num_faces_list = [0]
    num_vertices_list = [0]
    all_vertices_list = []
    all_faces_list = []
    all_face_normals_list = []

    for base_mesh in base_meshes:
        all_faces_list.append(base_mesh.faces + num_vertices_list[-1])
        all_vertices_list.append(base_mesh.vertices)
        num_faces_list.append(num_faces_list[-1] + base_mesh.faces.shape[0])
        num_vertices_list.append(num_vertices_list[-1] + base_mesh.vertices.shape[0])
        all_face_normals_list.append(base_mesh.face_normals)

    all_vertices = np.concatenate(all_vertices_list, axis=0)
    all_faces = np.concatenate(all_faces_list, axis=0)
    all_face_normals = np.concatenate(all_face_normals_list, axis=0)

    merged_mesh = trimesh.Trimesh(all_vertices, all_faces, process=False)

    locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(merged_mesh).intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )
    

    for mesh_i in range(len(base_meshes)):
        intersect_idxs = np.logical_and(index_tri >= num_faces_list[mesh_i], index_tri < num_faces_list[mesh_i + 1])
        contact_points = locations[intersect_idxs]
        contact_face_ids = index_tri[intersect_idxs] - num_faces_list[mesh_i]
        contact_mesh_ids = np.full(len(contact_points), mesh_i)
        contact_normals = all_face_normals[contact_face_ids]

        all_contact_points.append(contact_points)
        all_contact_mesh_ids.append(contact_mesh_ids)
        all_contact_face_ids.append(contact_face_ids)
        all_contact_face_normals.append(contact_normals)

    # Combine results from all base meshes
    if all_contact_points:
        contact_points = np.vstack(all_contact_points)
        contact_mesh_id = np.concatenate(all_contact_mesh_ids)
        contact_face_id = np.concatenate(all_contact_face_ids)
        contact_face_normals = np.vstack(all_contact_face_normals)
    else:
        # Return empty arrays if no contacts found
        contact_points = np.empty((0, 3), dtype=np.float32)
        contact_mesh_id = np.empty(0, dtype=np.int32)
        contact_face_id = np.empty(0, dtype=np.int32)
        contact_face_normals = np.empty((0, 3), dtype=np.float32)

    return contact_points, contact_mesh_id, contact_face_id, contact_face_normals



def falldown_collision_merged_max(base_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], test_mesh: Tuple[np.ndarray, np.ndarray]):
    # Convert test mesh to trimesh object
    test_vertices, test_faces = test_mesh
    test_trimesh = trimesh.Trimesh(vertices=test_vertices, faces=test_faces)
    

    # generate ray downs
    ray_origins = test_vertices
    ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(test_vertices.shape[0], axis=0)

    locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(test_trimesh).intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )
    average_edge_len = float(np.mean(np.linalg.norm(test_vertices[test_faces[:, 0]] - test_vertices[test_faces[:, 1]], axis=1)))

    index_ray_no_hit = np.ones(ray_origins.shape[0], dtype=np.bool_)
    index_ray_no_hit[index_ray] = False
    ray_origins = ray_origins[index_ray_no_hit]
    ray_directions = ray_directions[index_ray_no_hit]

    # Lists to store contact information
    all_contact_points = []
    all_contact_mesh_ids = []
    all_contact_face_ids = []
    all_contact_face_normals = []

    num_faces_list = [0]
    num_vertices_list = [0]
    all_vertices_list = []
    all_faces_list = []
    all_face_normals_list = []

    for base_mesh_triple in base_meshes:
        all_faces_list.append(base_mesh_triple[1] + num_vertices_list[-1])
        all_vertices_list.append(base_mesh_triple[0])
        num_faces_list.append(num_faces_list[-1] + base_mesh_triple[1].shape[0])
        num_vertices_list.append(num_vertices_list[-1] + base_mesh_triple[0].shape[0])
        all_face_normals_list.append(base_mesh_triple[2])

    all_vertices = np.concatenate(all_vertices_list, axis=0)
    all_faces = np.concatenate(all_faces_list, axis=0)
    all_face_normals = np.concatenate(all_face_normals_list, axis=0)

    merged_mesh = trimesh.Trimesh(all_vertices, all_faces, process=False)

    locations, index_ray, index_tri = trimesh.ray.ray_pyembree.RayMeshIntersector(merged_mesh).intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    max_intersect_mesh_id = -1
    max_intersect_face_cnt = 0

    distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
    

    for mesh_i in range(len(base_meshes)):
        intersect_idxs = np.logical_and(index_tri >= num_faces_list[mesh_i], index_tri < num_faces_list[mesh_i + 1])
        intersect_idxs = np.logical_and(intersect_idxs, distances <= 4.0 * average_edge_len)
        contact_points = locations[intersect_idxs]
        contact_face_ids = index_tri[intersect_idxs] - num_faces_list[mesh_i]
        contact_mesh_ids = np.full(len(contact_points), mesh_i)
        contact_normals = all_face_normals[contact_face_ids]

        if contact_points.shape[0] > max_intersect_face_cnt:
            max_intersect_face_cnt = contact_points.shape[0]
            max_intersect_mesh_id = mesh_i

            all_contact_points = [contact_points]
            all_contact_mesh_ids = [contact_mesh_ids]
            all_contact_face_ids = [contact_face_ids]
            all_contact_face_normals = [contact_normals]

    # Combine results from all base meshes
    if all_contact_points:
        contact_points = np.vstack(all_contact_points)
        contact_mesh_id = np.concatenate(all_contact_mesh_ids)
        contact_face_id = np.concatenate(all_contact_face_ids)
        contact_face_normals = np.vstack(all_contact_face_normals)
    else:
        # Return empty arrays if no contacts found
        contact_points = np.empty((0, 3), dtype=np.float32)
        contact_mesh_id = np.empty(0, dtype=np.int32)
        contact_face_id = np.empty(0, dtype=np.int32)
        contact_face_normals = np.empty((0, 3), dtype=np.float32)

    return contact_points, contact_mesh_id, contact_face_id, contact_face_normals






    
def marching_cubes_from_sdf_center_scale(model, center, scale, obj_id=None, resolution=512, level=0):
    assert obj_id is not None
    x = np.linspace(-1, 1, resolution) * scale + float(center.reshape(-1)[0])
    y = np.linspace(-1, 1, resolution) * scale + float(center.reshape(-1)[1])
    z = np.linspace(-1, 1, resolution) * scale + float(center.reshape(-1)[2])

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    with torch.no_grad():
        z_all = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z_all.append(model.get_sdf_raw(pnts.cuda())[:, obj_id].detach().cpu().numpy().reshape(-1))
        sdf_z = np.concatenate(z_all, axis=0)

        if (not (np.min(sdf_z) > level or np.max(sdf_z) < level)):
            sdf_z = sdf_z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=sdf_z.reshape(x.shape[0], y.shape[0], z.shape[0]),
                level=level,
                spacing=(x[2] - x[1],
                         y[2] - y[1],
                         z[2] - z[1]))

            verts = verts + np.array([x[0], y[0], z[0]])

            meshexport = trimesh.Trimesh(verts, faces, process=False)

            return meshexport
    return None

def marching_cubes_from_sdf_center_scale_rm_intersect(model, parent_sdfs, center, scale,
                                                      offset, scale_factor, collision_sample_cnt,
                                                      obj_id=None, resolution=512, level=0.00, prune_base_threshold_list=[-4, 0, 1, 2]):
    print("generate mesh from marching cubes...")
    assert obj_id is not None
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points_grid_idxs = torch.from_numpy(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T).float().reshape(-1, 3)
    points = points_grid_idxs * scale * scale_factor + torch.from_numpy(center).float().reshape(1, 3)

    points = points.reshape(resolution, resolution, resolution, 3)
    spacing_x = points[1, 0, 0, 0] - points[0, 0, 0, 0]
    spacing_y = points[0, 1, 0, 1] - points[0, 0, 0, 1]
    spacing_z = points[0, 0, 1, 2] - points[0, 0, 0, 2]

    volume_start_x = points[0, 0, 0, 0]
    volume_start_y = points[0, 0, 0, 1]
    volume_start_z = points[0, 0, 0, 2]
    points = points.reshape(-1, 3)

    points_grid_idxs = (points_grid_idxs * 0.5 + 0.5) * (collision_sample_cnt - 1)

    with torch.no_grad():
        z_all = []
        parent_z_all = []
        batch_size = 100000
        for i, (pnts, pnts_grid_idxs) in tqdm(enumerate(zip(torch.split(points, batch_size, dim=0), torch.split(points_grid_idxs, batch_size, dim=0))),
                                              total=int(np.ceil(points.shape[0] / batch_size))):
            sdfs = model.get_sdf_raw(pnts.cuda())[:, obj_id].detach().cpu().reshape(-1)
            psdfs = grid_sample_3d(parent_sdfs, pnts_grid_idxs)
            # prune_pts = torch.logical_and(sdfs < offset, psdfs < -4 * offset)
            # sdfs[prune_pts] = offset
            parent_z_all.append(psdfs.numpy())
            z_all.append(sdfs.numpy())

        sdf_z = np.concatenate(z_all, axis=0)
        parent_z = np.concatenate(parent_z_all, axis=0)

        sdf_z = sdf_z.astype(np.float32)
        parent_z = parent_z.astype(np.float32)

        prune_threshold_list = [base_threshold * offset for base_threshold in prune_base_threshold_list]
        export_mesh_list = []

        for prune_threshold in prune_threshold_list:

            prune_pts = np.logical_and(sdf_z < offset, parent_z < prune_threshold)
            sdf_z[prune_pts] = offset

            if (not (np.min(sdf_z) > level or np.max(sdf_z) < level)):

                verts, faces, normals, values = measure.marching_cubes(
                    volume=sdf_z.reshape(resolution, resolution, resolution),
                    level=level,
                    spacing=(spacing_x,
                             spacing_y,
                             spacing_z))

                verts = verts + np.array([volume_start_x, volume_start_y, volume_start_z])

                meshexport = trimesh.Trimesh(verts, faces, process=False)

            else:
                meshexport = None

            export_mesh_list.append(meshexport)

        return export_mesh_list

def marching_cubes_from_sdf_center_scale_single_object(model, center, scale, resolution=512, level=0):
    x = np.linspace(-1, 1, resolution) * scale + float(center.reshape(-1)[0])
    y = np.linspace(-1, 1, resolution) * scale + float(center.reshape(-1)[1])
    z = np.linspace(-1, 1, resolution) * scale + float(center.reshape(-1)[2])

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    with torch.no_grad():
        z_all = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z_all.append(model.get_sdf_vals(pnts.cuda()).detach().cpu().numpy().reshape(-1))
        sdf_z = np.concatenate(z_all, axis=0)

        if (not (np.min(sdf_z) > level or np.max(sdf_z) < level)):
            sdf_z = sdf_z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=sdf_z.reshape(x.shape[0], y.shape[0], z.shape[0]),
                level=level,
                spacing=(x[2] - x[1],
                         y[2] - y[1],
                         z[2] - z[1]))

            verts = verts + np.array([x[0], y[0], z[0]])

            meshexport = trimesh.Trimesh(verts, faces, process=False)

            return meshexport

def get_face_normals(verts, faces):
    triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    normals = np.cross(edge_01, edge_12)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals

def solve_intersection(meshes):
    # Create scene graph from meshes (parent-child relationships)
    from datasets.ns_dataset import extract_graph_node_properties
    parent_dict, child_dict = create_scene_graph_from_meshes(meshes)
    graph = convert_parent_child_to_adjacency_list(parent_dict, child_dict, len(meshes) - 1)
    graph_node_dict = extract_graph_node_properties(graph)
    num_objs = len(meshes)

    sim_mesh_list = [meshes[0]]
    sim_mesh_dict = {}
    sim_mesh_dict[0] = meshes[0]
    translation_dict = {}

    objs_seq_with_distance = [(obj_i, graph_node_dict[obj_i]['dist_to_root']) for obj_i in range(1, num_objs)]
    # sort objs_seq_with_distance by distance
    objs_seq_with_distance.sort(key=lambda x: x[1])
    objs_seq = [obj_i[0] for obj_i in objs_seq_with_distance]

    for _i, obj_i in enumerate(objs_seq):
        mesh_obj_i = meshes[obj_i]
        mesh_obj_i = trimesh.Trimesh(vertices=mesh_obj_i.vertices, faces=mesh_obj_i.faces, process=False)
        parent_id = graph_node_dict[obj_i]['parent']
        if parent_id != 0:
            base_translation = translation_dict[parent_id]
        else:
            base_translation = np.zeros(3).astype(np.float32).reshape(3)

        mesh_obj_i.vertices = mesh_obj_i.vertices + base_translation.reshape(1, 3)

        edge_length = np.mean(np.linalg.norm(mesh_obj_i.vertices[mesh_obj_i.faces[:, 0]] -
                                                mesh_obj_i.vertices[mesh_obj_i.faces[:, 1]],
                                                axis=1)) * 1.0
        print("solving intersection for obj: ", obj_i)
        for _ in range(20):
            contact_points, contact_mesh_id, contact_face_id, contact_face_normals = \
                detect_collision(
                    [(mesh.vertices, mesh.faces, mesh.face_normals) for i, mesh in enumerate(sim_mesh_list)],
                    (mesh_obj_i.vertices, mesh_obj_i.faces))

            if len(contact_points) == 0:
                break

            average_normals = np.mean(contact_face_normals, axis=0)
            average_normals = average_normals / np.linalg.norm(average_normals)

            average_normals = average_normals.reshape(1, 3)

            mesh_obj_i.vertices = mesh_obj_i.vertices + average_normals * edge_length
            base_translation = base_translation + average_normals * edge_length

        sim_mesh_list.append(mesh_obj_i)
        translation_dict[obj_i] = base_translation
        sim_mesh_dict[obj_i] = mesh_obj_i

        print("finish solving intersection for obj: ", obj_i)
        print("processing: ", f"{_i+1} / {len(objs_seq)}")

    return [sim_mesh_dict[obj_i] for obj_i in range(num_objs) if obj_i in sim_mesh_dict.keys()]

def calculate_adjacency_matrices_from_meshes(all_meshes, support_normal_threshold=0.75):
    """
    Calculate collision, support, and desupport adjacency matrices from a list of meshes.
    
    Args:
        all_meshes (list): List of trimesh objects
        support_normal_threshold (float): Threshold for determining support vs desupport based on face normal z-component
        
    Returns:
        tuple: (collision_adjacency_matrix, support_adjacency_matrix, desupport_adjacency_matrix)
            - collision_adjacency_matrix: Binary matrix indicating collisions between objects
            - support_adjacency_matrix: Normalized matrix indicating support relationships
            - desupport_adjacency_matrix: Normalized matrix indicating desupport (hanging) relationships
    """
    total_num_objs = len(all_meshes) - 1  # Assuming last object is background/room
    
    collision_adjacency_matrix = np.zeros((total_num_objs+1, total_num_objs+1))
    support_adjacency_matrix = np.zeros((total_num_objs+1, total_num_objs+1))
    desupport_adjacency_matrix = np.zeros((total_num_objs+1, total_num_objs+1))
    
    for instance_id in range(total_num_objs+1):
        test_mesh = all_meshes[instance_id]
        base_meshes = [all_meshes[obj_i] for obj_i in range(total_num_objs+1) if obj_i != instance_id]
        base_meshes_info = [(mesh.vertices, mesh.faces, mesh.face_normals) for mesh in base_meshes]
        test_mesh_info = (test_mesh.vertices, test_mesh.faces)

        contact_points, contact_mesh_id, contact_face_id, contact_face_normals = detect_collision(base_meshes_info, test_mesh_info)
        if contact_points.shape[0] == 0:
            contact_points, contact_mesh_id, contact_face_id, contact_face_normals = falldown_collision(base_meshes_info, test_mesh_info)
            if contact_points.shape[0] == 0:
                continue

        contact_mesh_id[contact_mesh_id >= instance_id] += 1

        for obj_i in range(total_num_objs+1):
            if obj_i == instance_id:
                continue
            collided = contact_mesh_id == obj_i
            if np.any(collided):
                collision_adjacency_matrix[instance_id, obj_i] = 1
                contact_face_normals_obj_i = contact_face_normals[collided]
                support_cnt = np.count_nonzero(contact_face_normals_obj_i[:, 2] > support_normal_threshold)
                desupport_cnt = np.count_nonzero(contact_face_normals_obj_i[:, 2] < -support_normal_threshold)
                support_adjacency_matrix[instance_id, obj_i] += support_cnt
                desupport_adjacency_matrix[instance_id, obj_i] += desupport_cnt

                collision_adjacency_matrix[obj_i, instance_id] = 1
                contact_face_normals_obj_i_inv = -contact_face_normals_obj_i
                support_cnt = np.count_nonzero(contact_face_normals_obj_i_inv[:, 2] > support_normal_threshold)
                desupport_cnt = np.count_nonzero(contact_face_normals_obj_i_inv[:, 2] < -support_normal_threshold)
                support_adjacency_matrix[obj_i, instance_id] += support_cnt
                desupport_adjacency_matrix[obj_i, instance_id] += desupport_cnt
        
    # Normalize support and desupport matrices
    # Handle division by zero - replace NaN with 0
    support_sum = support_adjacency_matrix.sum(axis=1, keepdims=True)
    support_sum[support_sum == 0] = 1  # Avoid division by zero
    support_adjacency_matrix = support_adjacency_matrix / support_sum
    
    desupport_sum = desupport_adjacency_matrix.sum(axis=1, keepdims=True)
    desupport_sum[desupport_sum == 0] = 1  # Avoid division by zero
    desupport_adjacency_matrix = desupport_adjacency_matrix / desupport_sum
    
    # Replace any remaining NaN values with 0
    support_adjacency_matrix = np.nan_to_num(support_adjacency_matrix)
    desupport_adjacency_matrix = np.nan_to_num(desupport_adjacency_matrix)
    
    return collision_adjacency_matrix, support_adjacency_matrix, desupport_adjacency_matrix


def calculate_adjacency_matrices_falldown(all_meshes, support_normal_threshold=0.75):
    """
    Calculate collision, support, and desupport adjacency matrices from a list of meshes using falldown collision.
    
    Args:
        all_meshes (list): List of trimesh objects
        support_normal_threshold (float): Threshold for determining support vs desupport based on face normal z-component
        
    Returns:
        tuple: (collision_adjacency_matrix, support_adjacency_matrix, desupport_adjacency_matrix)
    """
    total_num_objs = len(all_meshes) - 1  # Assuming last object is background/room
    
    collision_adjacency_matrix = np.zeros((total_num_objs+1, total_num_objs+1))
    support_adjacency_matrix = np.zeros((total_num_objs+1, total_num_objs+1))
    desupport_adjacency_matrix = np.zeros((total_num_objs+1, total_num_objs+1))
    
    for instance_id in range(total_num_objs+1):
        test_mesh = all_meshes[instance_id]
        base_meshes = [all_meshes[obj_i] for obj_i in range(total_num_objs+1) if obj_i != instance_id]
        base_meshes_info = [(mesh.vertices, mesh.faces, mesh.face_normals) for mesh in base_meshes]
        test_mesh_info = (test_mesh.vertices, test_mesh.faces)

        contact_points, contact_mesh_id, contact_face_id, contact_face_normals = falldown_collision_merged_max(base_meshes_info, test_mesh_info)

        contact_mesh_id[contact_mesh_id >= instance_id] += 1

        for obj_i in range(total_num_objs+1):
            if obj_i == instance_id:
                continue
            collided = contact_mesh_id == obj_i
            if np.any(collided):
                collision_adjacency_matrix[instance_id, obj_i] = 1
                contact_face_normals_obj_i = contact_face_normals[collided]
                support_cnt = np.count_nonzero(contact_face_normals_obj_i[:, 2] > support_normal_threshold)
                desupport_cnt = np.count_nonzero(contact_face_normals_obj_i[:, 2] < -support_normal_threshold)
                support_adjacency_matrix[instance_id, obj_i] += support_cnt
                desupport_adjacency_matrix[instance_id, obj_i] += desupport_cnt

                collision_adjacency_matrix[obj_i, instance_id] = 1
                contact_face_normals_obj_i_inv = -contact_face_normals_obj_i
                support_cnt = np.count_nonzero(contact_face_normals_obj_i_inv[:, 2] > support_normal_threshold)
                desupport_cnt = np.count_nonzero(contact_face_normals_obj_i_inv[:, 2] < -support_normal_threshold)
                support_adjacency_matrix[obj_i, instance_id] += support_cnt
                desupport_adjacency_matrix[obj_i, instance_id] += desupport_cnt

    # Normalize support and desupport matrices
    # Handle division by zero - replace NaN with 0
    support_sum = support_adjacency_matrix.sum(axis=1, keepdims=True)
    support_sum[support_sum == 0] = 1  # Avoid division by zero
    support_adjacency_matrix = support_adjacency_matrix / support_sum
    
    desupport_sum = desupport_adjacency_matrix.sum(axis=1, keepdims=True)
    desupport_sum[desupport_sum == 0] = 1  # Avoid division by zero
    desupport_adjacency_matrix = desupport_adjacency_matrix / desupport_sum
    
    # Replace any remaining NaN values with 0
    support_adjacency_matrix = np.nan_to_num(support_adjacency_matrix)
    desupport_adjacency_matrix = np.nan_to_num(desupport_adjacency_matrix)
    
    return collision_adjacency_matrix, support_adjacency_matrix, desupport_adjacency_matrix


def get_all_descendants_graph(node, child_dict):
    """
    Get all descendants of a node in a tree recursively
    
    Args:
        node: The node ID to find descendants for
        child_dict: Dictionary mapping parent node IDs to lists of direct child node IDs
        
    Returns:
        list: All descendant node IDs (children, grandchildren, etc.)
    """
    descendants = []
    
    # Get direct children of this node
    direct_children = child_dict.get(node, [])
    
    # Add all direct children to descendants
    descendants.extend(direct_children)
    
    # Recursively get descendants of each direct child
    for child in direct_children:
        descendants.extend(get_all_descendants_graph(child, child_dict))
    
    return descendants


def create_scene_graph_from_meshes(all_meshes, support_normal_threshold=0.90):
    """
    Create a scene graph from a list of meshes by calculating parent-child relationships.
    
    Args:
        all_meshes (list): List of trimesh objects
        support_normal_threshold (float): Threshold for determining support relationships
        
    Returns:
        tuple: (parent_dict, child_dict) representing the scene graph
    """
    from random import shuffle
    
    total_num_objs = len(all_meshes) - 1
    
    # Calculate adjacency matrices
    collision_matrix, support_matrix, desupport_matrix = calculate_adjacency_matrices_from_meshes(
        all_meshes, support_normal_threshold
    )
    
    collision_matrix_falldown, support_matrix_falldown, desupport_matrix_falldown = calculate_adjacency_matrices_falldown(
        all_meshes, support_normal_threshold
    )
    
    # Create a parent-child relationship storage dict
    parent_dict = {}
    child_dict = {}
    floor_object_idxs = []

    # Find all floor objects: has collision with the bg mesh (idx=0) and the argmax of support matrix is 0
    # The parent of the floor objects is the bg mesh
    ground_collision_idxs = np.nonzero(collision_matrix_falldown[:, 0])[0]

    for instance_id in range(1, total_num_objs+1):
        if collision_matrix_falldown[0, instance_id] == 1 and np.argmax(support_matrix_falldown[instance_id, [0]+ground_collision_idxs.tolist()]) == 0:
            parent_dict[instance_id] = 0
            child_dict[0] = child_dict.get(0, []) + [instance_id]
            floor_object_idxs.append(instance_id)

    # Find leaf objects idx: for all collided objects obj_i, support_matrix[idx, obj_i] > support_matrix[obj_i, idx]
    leaf_object_idxs = []
    for instance_id in range(1, total_num_objs+1):
        if instance_id in parent_dict:
            continue
        collided_idxs = np.nonzero(collision_matrix[instance_id])[0]

        leaf = True
        for collided_idx in collided_idxs:
            if support_matrix[instance_id, collided_idx] < support_matrix[collided_idx, instance_id]:
                leaf = False
                break
        if leaf:
            leaf_object_idxs.append(instance_id)

    while len(leaf_object_idxs) > 0:
        to_add_idx = leaf_object_idxs.pop(0)
        if to_add_idx in parent_dict:
            continue
        support_sorted_idxs = np.argsort(support_matrix[to_add_idx])[::-1]
        collided_idxs = collision_matrix[to_add_idx]
        all_desc_idxs = get_all_descendants_graph(to_add_idx, child_dict)

        support_sorted_idxs = support_sorted_idxs[collided_idxs[support_sorted_idxs].astype(np.bool_)].tolist()
        support_sorted_idxs = [idx for idx in support_sorted_idxs if idx not in all_desc_idxs and idx != 0]

        if len(support_sorted_idxs) == 0:
            continue
            
        select_parent_idx = 0
        parent_idx = support_sorted_idxs[select_parent_idx]
        
        while parent_idx in floor_object_idxs and select_parent_idx + 1 < len(support_sorted_idxs):
            select_parent_idx += 1
            parent_idx = support_sorted_idxs[select_parent_idx]
        
        parent_dict[to_add_idx] = parent_idx
        child_dict[parent_idx] = child_dict.get(parent_idx, []) + [to_add_idx]
        if parent_idx not in leaf_object_idxs:
            leaf_object_idxs.append(parent_idx)

    remaining_object_idxs = [obj_i for obj_i in range(1, total_num_objs+1) if obj_i not in parent_dict]
    shuffle(remaining_object_idxs)
    for to_add_idx in remaining_object_idxs:
        support_sorted_idxs = np.argsort(support_matrix[to_add_idx])[::-1]
        collided_idxs = collision_matrix[to_add_idx]
        all_desc_idxs = get_all_descendants_graph(to_add_idx, child_dict)

        support_sorted_idxs = support_sorted_idxs[collided_idxs[support_sorted_idxs].astype(np.bool_)].tolist()
        support_sorted_idxs = [idx for idx in support_sorted_idxs if idx not in all_desc_idxs and idx != 0]

        if len(support_sorted_idxs) == 0:
            continue
            
        select_parent_idx = 0
        parent_idx = support_sorted_idxs[select_parent_idx]
        
        while parent_idx in floor_object_idxs and select_parent_idx + 1 < len(support_sorted_idxs):
            select_parent_idx += 1
            parent_idx = support_sorted_idxs[select_parent_idx]
        
        parent_dict[to_add_idx] = parent_idx
        child_dict[parent_idx] = child_dict.get(parent_idx, []) + [to_add_idx]

    return parent_dict, child_dict


def convert_parent_child_to_adjacency_list(parent_dict, child_dict, total_num_objs):
    """
    Convert parent-child dictionaries to adjacency list format like graph.json.
    
    Args:
        parent_dict (dict): Dictionary mapping child nodes to their parents
        child_dict (dict): Dictionary mapping parent nodes to lists of children
        total_num_objs (int): Total number of objects (excluding background)
        
    Returns:
        list: List of dictionaries in the format [{"node_id": id, "adj_nodes": [list]}]
    """
    graph = []
    
    for node_id in range(total_num_objs + 1):  # Include background (node 0)
        adj_nodes = []
        
        # Add parent if exists
        if node_id in parent_dict:
            adj_nodes.append(parent_dict[node_id])
        
        # Add children if exist
        if node_id in child_dict:
            adj_nodes.extend(child_dict[node_id])
        
        graph.append({
            "node_id": node_id,
            "adj_nodes": adj_nodes
        })
    
    return graph



def generate_color_from_model_and_mesh(model, mesh, idx):
    verts = mesh.vertices
    faces = mesh.faces

    face_normals = get_face_normals(verts, faces)
    mean_face_dist = float(np.mean(np.linalg.norm(verts[faces[:, 0]] - verts[faces[:, 1]], axis=-1)))


    face_centroids = np.mean(verts[faces.reshape(-1)].reshape(-1, 3, 3), axis=1).reshape(-1, 3)
    color_all = []
    for i, (vert_pnts, vert_normal) in enumerate(
            zip(torch.split(torch.from_numpy(face_centroids.reshape(-1, 3).copy()), 1024, dim=0),
                torch.split(torch.from_numpy(face_normals.reshape(-1, 3).copy()), 1024, dim=0))):
        ray_start = vert_pnts + vert_normal * mean_face_dist * 0.2
        ray_dir = -vert_normal

        color = model.get_colors_from_point_rays_obj_offset(
            ray_start.cuda().float().detach(),
            ray_dir.cuda().float().detach(),
            idx).detach().cpu().numpy()
        color_all.append(color)
        torch.cuda.empty_cache()
    face_color = np.concatenate(color_all, axis=0)
    # face_color = np.clip(face_color * 255., 0, 255)
    # print("face_color 2: ", face_color.shape, face_color.max(), face_color.min())
    meshexport = trimesh.Trimesh(verts, faces, face_colors=face_color, process=False)

    return meshexport