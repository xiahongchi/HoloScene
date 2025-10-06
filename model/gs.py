# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch.nn as nn
import numpy as np
import torch
import trimesh
import cv2

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
# from gsplat.cuda_legacy._wrapper import num_sh_bases

def num_sh_bases(degree):
     if degree == 0:
         return 1
     if degree == 1:
         return 4
     if degree == 2:
         return 9
     if degree == 3:
         return 16
     return 25

from typing import Literal


from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    euler_angles_to_matrix,
)
from pytorch3d.io import load_objs_as_meshes, save_obj
from collections import namedtuple
from pytorch3d.renderer import TexturesUV, TexturesVertex, RasterizationSettings, MeshRenderer, MeshRendererWithFragments, MeshRasterizer, HardPhongShader, PointLights
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F_V

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def area(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    return area

def circumcircle_radius(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    # Compute the circumcircle radius
    R = (a * b * c) / (4 * area)

    return R

def incircle_radius(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    # Compute the circumcircle radius
    R = area / s

    return R


def compute_min_distance(v, v1, v2, v3):
    """
    Computes the minimum distance from point v to the edges of the triangle formed by v1, v2, and v3.
    Supports batch inference.

    Args:
    v (Tensor): Tensor of shape (batch_size, 3), the point from which distances are computed.
    v1, v2, v3 (Tensor): Tensors of shape (batch_size, 3), representing the vertices of the triangle.

    Returns:
    Tensor: Minimum distance from v to the triangle edges for each batch element.
    """

    def distance_to_edge(v, v_start, v_end):
        # Compute edge vector e and vector from start to v (w)
        e = v_end - v_start
        w = v - v_start

        # Projection scalar t
        e_dot_e = torch.sum(e * e, dim=-1, keepdim=True)
        t = torch.sum(w * e, dim=-1, keepdim=True) / e_dot_e

        # Clamp t to [0, 1] to handle closest point on the edge segment
        t_clamped = torch.clamp(t, 0, 1)

        # Closest point on the edge
        closest_point = v_start + t_clamped * e

        # Compute distance from v to closest point
        distance = torch.norm(v - closest_point, dim=-1)

        return distance

    # Compute distances to each edge
    d1 = distance_to_edge(v, v1, v2)  # Edge from v1 to v2
    d2 = distance_to_edge(v, v2, v3)  # Edge from v2 to v3
    d3 = distance_to_edge(v, v3, v1)  # Edge from v3 to v1

    # Return the minimum distance across the edges
    min_distance = torch.min(torch.stack([d1, d2, d3], dim=-1), dim=-1).values

    return min_distance


def compute_triangle_vertices(a, b, c):
    """
    Compute the coordinates of triangle vertices A, B, and C
    given side lengths a, b, and c in a batched manner.

    Args:
        a: Tensor of side lengths |BC|.
        b: Tensor of side lengths |CA|.
        c: Tensor of side lengths |AB|.

    Returns:
        A (0, 0), B (c, 0), C (x_C, y_C): Coordinates of the triangle vertices.
    """
    # Vertices A and B
    A_x, A_y = torch.zeros_like(a), torch.zeros_like(a)  # A is at (0, 0)
    B_x, B_y = c, torch.zeros_like(c)  # B is at (c, 0)

    # Compute C coordinates
    x_C = (c ** 2 - a ** 2 + b ** 2) / (2 * c)  # x-coordinate of C
    y_C = torch.sqrt(b ** 2 - x_C ** 2)  # y-coordinate of C

    A = torch.stack((A_x, A_y), dim=-1)
    B = torch.stack((B_x, B_y), dim=-1)
    C = torch.stack((x_C, y_C), dim=-1)

    return A, B, C


def barycentric_coordinates(P, A, B, C):
    """
    Compute the barycentric coordinates for a point P relative to triangle ABC.

    Args:
        P_x: Tensor of x-coordinates of point P.
        P_y: Tensor of y-coordinates of point P.
        A, B, C: Tuples of tensors representing the coordinates of points A, B, and C.

    Returns:
        alpha, beta, gamma: Barycentric coordinates of the point P.
    """
    A_x, A_y = A[:, 0], A[:, 1]
    B_x, B_y = B[:, 0], B[:, 1]
    C_x, C_y = C[:, 0], C[:, 1]
    P_x, P_y = P[:, 0], P[:, 1]

    # Compute the denominator (area of the triangle ABC)
    By_Cy = B_y - C_y
    Ax_Cx = A_x - C_x
    Cx_Bx = C_x - B_x
    Ay_Cy = A_y - C_y
    Px_Cx = P_x - C_x
    Py_Cy = P_y - C_y

    denominator = By_Cy * Ax_Cx + Cx_Bx * Ay_Cy

    # Compute the barycentric coordinates
    alpha = (By_Cy * Px_Cx + Cx_Bx * Py_Cy) / denominator
    beta = (-Ay_Cy * Px_Cx + Ax_Cx * Py_Cy) / denominator
    gamma = 1 - alpha - beta

    return torch.stack([alpha, beta, gamma], dim=-1)

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    height, width = image.shape[:2]
    # print("height: ", height)
    # print("width: ", width)

    scaling_factor = 1.0 / d
    # weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    # return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)
    height_new = int(torch.floor(0.5 + (torch.tensor([float(height)]) * scaling_factor)).to(torch.int64))
    width_new = int(torch.floor(0.5 + (torch.tensor([float(width)]) * scaling_factor)).to(torch.int64))

    # print("height_new: ", height_new)
    # print("width_new: ", width_new)

    image_resized = F_V.resize(image.permute(2, 0, 1), [height_new, width_new])
    image_resized = image_resized.permute(1, 2, 0)

    return image_resized

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    optimized_camera_to_world = optimized_camera_to_world.reshape(1, -1, 4)
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    # R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

def write_ply(
    filename: str,
    count: int,
    map_to_tensors,
):
    """
    Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
    Note: All float values will be converted to float32 for writing.

    Parameters:
    filename (str): The name of the file to write.
    count (int): The number of vertices to write.
    map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
        Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
    """

    # Ensure count matches the length of all tensors
    if not all(len(tensor) == count for tensor in map_to_tensors.values()):
        raise ValueError("Count does not match the length of all tensors")

    # Type check for numpy arrays of type float or uint8 and non-empty
    if not all(
        isinstance(tensor, np.ndarray)
        and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
        and tensor.size > 0
        for tensor in map_to_tensors.values()
    ):
        raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

    with open(filename, "wb") as ply_file:
        # Write PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")

        ply_file.write(f"element vertex {count}\n".encode())

        # Write properties, in order due to OrderedDict
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())

        ply_file.write(b"end_header\n")

        # Write binary data
        # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())
from collections import OrderedDict

def compose_for_export(model: GS):
    means_drawer_i = model.means.detach().clone().cpu().numpy()
    shs_0_drawer_i = model.shs_0().detach().clone().contiguous().cpu().numpy()
    shs_rest_drawer_i = model.shs_rest().detach().clone().transpose(1, 2).cpu().numpy()
    colors_drawer_i = torch.clamp(model.colors().detach().clone(), 0.0, 1.0).data.cpu().numpy()
    opacities_drawer_i = model.opacities.detach().clone().cpu().numpy()
    scales_drawer_i = model.scales.detach().clone().cpu().numpy()
    quats_drawer_i = model.quats.detach().clone().cpu().numpy()

    map_to_tensors = OrderedDict()

    with torch.no_grad():

        positions = means_drawer_i
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        if model.config.sh_degree > 0:
            shs_0 = shs_0_drawer_i
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            # transpose(1, 2) was needed to match the sh order in Inria version
            shs_rest = shs_rest_drawer_i
            shs_rest = shs_rest.reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
        else:
            colors = colors_drawer_i
            map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

        map_to_tensors["opacity"] = opacities_drawer_i

        scales = scales_drawer_i
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = quats_drawer_i
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        n_before = np.sum(select)
        if len(t.shape) > 1:
            finite = np.isfinite(t).all(axis=-1)
        else:
            finite = np.isfinite(t)
        select = np.logical_and(select, finite)
        n_after = np.sum(select)
        if n_after < n_before:
            print(f"{n_before - n_after} NaN/Inf elements in {k}")
            print(t.shape, t)

    if np.sum(select) < n:
        print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
        for k, t in map_to_tensors.items():
            map_to_tensors[k] = map_to_tensors[k][select]
        count = np.sum(select)

    return map_to_tensors, count


@dataclass
class SplatfactoOnMeshUCModelConfig:
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 1
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    """Config of the camera optimizer to use"""
    mesh_area_to_subdivide: float = 2e-6
    upper_scale: float = 2.0
    unconstrained_scale: bool = True
    unconstrained_elevate: bool = True
    face_flat_coef: float = 0.005
    elevate_coef: float = 2.0
    cone_coef: float = 10.0 * np.pi / 180.0
    acm_lambda: float = 20.0
    mesh_depth_lambda: float = 10.0
    gaussian_save_extra_info_path: str = None

def load_3D_points_on_mesh(texture_mesh_path):
    obj_filename = texture_mesh_path

    ext = os.path.splitext(obj_filename)[-1].lower()

    if ext == ".obj":

        mesh = load_objs_as_meshes([obj_filename], device='cpu')
        # while mesh.faces_packed().shape[0] < target_pts_num:
        #     mesh = SubdivideMeshes()(mesh)
        #     print("face number: ", mesh.faces_packed().shape[0])

        mesh_verts = mesh.verts_packed().clone().reshape(-1, 3)
        mesh_faces = mesh.faces_packed().clone().reshape(-1, 3)

        N_Gaussians = mesh_faces.shape[0]
        triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)
        means = torch.mean(triangles, dim=1)
        radius = circumcircle_radius(triangles)

        Mesh_Fragments = namedtuple("Mesh_Fragments", ['pix_to_face', 'bary_coords'])
        mesh_fragments = Mesh_Fragments(
            pix_to_face=torch.arange(N_Gaussians).reshape(1, 1, N_Gaussians, 1),
            bary_coords=(torch.ones(1, 1, N_Gaussians, 1, 3) / 3)
        )
        features_dc = mesh.textures.sample_textures(mesh_fragments).reshape(N_Gaussians, 3)

        normals = mesh.faces_normals_packed().clone().reshape(-1, 3)

    elif ext == ".ply":
        mesh = trimesh.load(obj_filename, process=False)
        mesh_verts = torch.from_numpy(mesh.vertices).float()
        mesh_faces = torch.from_numpy(mesh.faces).long()
        N_Gaussians = mesh_faces.shape[0]
        triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)
        means = torch.mean(triangles, dim=1)
        radius = circumcircle_radius(triangles)
        normals = torch.from_numpy(mesh.face_normals).float().reshape(-1, 3)
        features_dc = torch.from_numpy(mesh.visual.face_colors).float().reshape(N_Gaussians, -1)[:, :3] / 255.

    area_to_subdivide = 2e-6

    print("before subdivision: ")
    print("num of gs points: ", N_Gaussians)
    while True:
        areas = area(triangles)
        if torch.all(areas[torch.isfinite(areas)] <= area_to_subdivide):
            break
        face_to_subdivide = (areas > area_to_subdivide)

        mesh_faces_subdivided = mesh_faces[face_to_subdivide]

        verts_0_idxs = mesh_faces_subdivided[:, 0]
        verts_1_idxs = mesh_faces_subdivided[:, 1]
        verts_2_idxs = mesh_faces_subdivided[:, 2]

        # edges_faces_subdivided = mesh_faces_subdivided[:, 0]
        edges_faces_subdivided_01 = torch.stack([verts_0_idxs, verts_1_idxs], dim=-1)
        edges_faces_subdivided_02 = torch.stack([verts_0_idxs, verts_2_idxs], dim=-1)
        edges_faces_subdivided_12 = torch.stack([verts_1_idxs, verts_2_idxs], dim=-1)

        edges_faces_subdivided_01 = torch.sort(edges_faces_subdivided_01, dim=-1)[0]
        edges_faces_subdivided_02 = torch.sort(edges_faces_subdivided_02, dim=-1)[0]
        edges_faces_subdivided_12 = torch.sort(edges_faces_subdivided_12, dim=-1)[0]


        edges_faces_subdivided = torch.stack([
            edges_faces_subdivided_01,
            edges_faces_subdivided_02,
            edges_faces_subdivided_12,
        ], dim=1)
        edges_faces_subdivided_flatten = edges_faces_subdivided.reshape(-1, 2)
        edges_faces_subdivided_unique, edges_faces_subdivided_unique_inverse_idx = torch.unique(edges_faces_subdivided_flatten, return_inverse=True, dim=0)

        edges_faces_subdivided_unique_verts = mesh_verts[edges_faces_subdivided_unique.reshape(-1)].reshape(-1, 2, 3)
        mesh_verts_added = (edges_faces_subdivided_unique_verts[:, 0] + edges_faces_subdivided_unique_verts[:, 1]) / 2

        num_verts_before = mesh_verts.shape[0]
        mesh_verts_added_idxs = num_verts_before + torch.arange(mesh_verts_added.shape[0])

        verts_abc_idxs = mesh_verts_added_idxs[edges_faces_subdivided_unique_inverse_idx]
        verts_abc_idxs = verts_abc_idxs.reshape(-1, 3)

        verts_a_idxs = verts_abc_idxs[:, 0]
        verts_b_idxs = verts_abc_idxs[:, 1]
        verts_c_idxs = verts_abc_idxs[:, 2]

        faces_0ab = torch.stack([verts_0_idxs, verts_a_idxs, verts_b_idxs], dim=-1)
        faces_1ca = torch.stack([verts_1_idxs, verts_c_idxs, verts_a_idxs], dim=-1)
        faces_2bc = torch.stack([verts_2_idxs, verts_b_idxs, verts_c_idxs], dim=-1)
        faces_acb = torch.stack([verts_a_idxs, verts_c_idxs, verts_b_idxs], dim=-1)

        mesh_faces[face_to_subdivide] = faces_acb
        mesh_faces = torch.cat([
            mesh_faces,
            faces_0ab, faces_1ca, faces_2bc
        ], dim=0)

        mesh_verts = torch.cat([
            mesh_verts,
            mesh_verts_added
        ], dim=0)

        triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)

        radius = circumcircle_radius(triangles)
        N_Gaussians = mesh_faces.shape[0]
        means = torch.mean(triangles, dim=1)
        normals = torch.cat([normals] + [normals[face_to_subdivide]] * 3, dim=0)
        features_dc = torch.cat([features_dc] + [features_dc[face_to_subdivide]] * 3, dim=0)
    print("after basic subdivision: ")
    print("num of gs points: ", N_Gaussians)


    out = {
        "means": means,
        "radius": radius,
        "features_dc": features_dc,
        "normals": normals,
        "mesh_verts": mesh_verts,
        "mesh_faces": mesh_faces,
    }

    return out

class GS(nn.Module):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoOnMeshUCModelConfig

    def __init__(
        self,
        config = SplatfactoOnMeshUCModelConfig(),
        seed_gs = None,
    ):
        super().__init__()
        self.config = config
        self.seed_gs = seed_gs
        assert self.seed_gs is not None, "need gs seed"
        self.populate_modules()

    def populate_modules(self):
        self.means = self.seed_gs["means"].cuda().float()
        self.opacities = self.seed_gs["opacities"].cuda().float()
        self.features_dc = self.seed_gs["features_dc"].cuda().float()
        self.features_rest = self.seed_gs["features_rest"].cuda().float()
        self.scales = self.seed_gs["scales"].cuda().float()
        self.quats = self.seed_gs["quats"].cuda().float()
        self.config.sh_degree = self.seed_gs["sh_degree"]

        self.background_color = torch.tensor(
            [0.1490, 0.1647, 0.2157]
        )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.

    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    def shs_0(self):
        return self.features_dc

    def shs_rest(self):
        return self.features_rest

    def num_points(self):
        return self.means.shape[0]

    def export_gs(self, export_path):
        map_to_tensors, count = compose_for_export(self)
        write_ply(export_path, count, map_to_tensors)

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def _get_downscale_factor(self):
        return 1

    def _downscale_if_required(self, image):
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, pose, K, H, W, camera_model) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        means_crop = self.means.float()
        opacities_crop = self.opacities.float()
        features_dc_crop = self.features_dc.float()
        features_rest_crop = self.features_rest.float()
        scales_crop = self.scales.float()
        quats_crop = self.quats.float()

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        viewmat = get_viewmat(pose).cuda().float()
        K = K.cuda().reshape(1, 3, 3).float()
        W, H = int(W), int(H)
        self.last_size = (H, W)

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        render_mode = "RGB+ED"

        if self.config.sh_degree > 0:
            sh_degree_to_use = self.config.sh_degree
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            camera_model=camera_model
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

