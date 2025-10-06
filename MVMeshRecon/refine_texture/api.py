import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.utils import save_image
import trimesh
from trimesh.visual import TextureVisuals
import open3d as o3d

from .refine_texture.common.cameras.mvp_matrix_v2 import c2w_to_w2c, proj_to_intr
from .refine_texture.common.mesh.structure import Mesh, Texture
from .refine_texture.mesh_remapping import remapping_vertex_color, remapping_uv_texture, initial_map_Kd_with_v_rgb
# from utils.general_utils import erode_mask
def erode_mask(mask: torch.Tensor, kernel_size=5):
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    mask_erode_list = []
    for mask_i in mask_np:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_np_erode = cv2.erode(mask_i, kernel=kernel)
        mask_erode = torch.Tensor(mask_np_erode).to(mask) / 255
        mask_erode_list.append(mask_erode[None, ..., None])
    mask_new = torch.concat(mask_erode_list, dim=0)

    return mask_new

def save_trimesh(mesh, v_colors, save_path):
    v_colors = v_colors.detach().cpu().numpy()
    visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=v_colors)
    mesh.visual = visual
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    mesh.export(save_path)
    

def opt_warpper(
    vertices,
    faces,
    images,
    w2cs,
    projections,
    refine_index,
    weights,
    visualize=False,
    render_size=512,
    b_perspective=True,
    do_refine_uv=True,
):
    '''
    vertices: [V, 3]
    faces: [F, 3]
    images: [N, H, W, C]
    c2ws: [N, 4, 4]
    projections: [N, 4, 4] or [4, 4]
    intrinsics: [N, 3, 3] or [3, 3], normalized

    uvs: [V, 2]
    map_Kd: [H, W, C]
    '''
    # build dataset
    if images.shape[-1] != 4:
        # rembg_session = rembg.new_session()
        images_np = images.clamp(0, 1).mul(255.0).detach().cpu().numpy().astype(np.uint8)
        _images = torch.empty((*images.shape[:-1], 4), dtype=images.dtype)
        for i, im in enumerate(images_np):
            # im = rembg.remove(im, alpha_matting=True, session=rembg_session)
            _im = cv2.inpaint(im[:, :, :3], ~im[:, :, [3]], 3.0, cv2.INPAINT_TELEA)
            im = np.concatenate([_im, im[:, :, [3]]], axis=-1)
            _images[i] = torch.as_tensor(im, dtype=images.dtype).div(255.0)
        images = _images.to(device=images.device)

    images[..., [3]] = erode_mask((images[..., [3]]>0.5).float())
    # save_image(images.permute(0,3,1,2), "debug/RGBs_rmbg.png")
    images = images.permute(0, 3, 1, 2)
    dataset_dict = {}
    c2ws = c2w_to_w2c(w2cs)
    intrinsics = proj_to_intr(projections.expand(c2ws.shape[0], 4, 4), perspective=b_perspective)

    dataset_dict['frames_c2w'] = c2ws
    dataset_dict['intrinsic'] = intrinsics.expand(c2ws.shape[0], 3, 3)
    dataset_dict['frames_img'] = images
    dataset_dict = {k: v.to(device='cuda') if isinstance(v, torch.Tensor) else v for k, v in dataset_dict.items()}

    # load mesh
    # print("opt warpper:")
    # print("1 vertices:", vertices.shape)
    # print("1 faces:", faces.shape)
    mesh = Mesh(vertices, faces).to_open3d()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 500000:
        # mesh = mesh.simplify_quadric_decimation(200_000)
        device_o3d = o3d.core.Device('CPU:0')
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=device_o3d)
        # target_reduction = 1 - 200_000 / len(mesh.triangle.indices)
        target_reduction = 1-100000 / len(mesh.triangle.indices)
        mesh = mesh.simplify_quadric_decimation(target_reduction)
        mesh = mesh.to_legacy()
        mesh = mesh.remove_non_manifold_edges()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
    mesh = Mesh.from_open3d(mesh).to_trimesh()
    if not isinstance(mesh.visual, TextureVisuals):
        mesh.visual = TextureVisuals(image=Image.fromarray(np.full((2, 2, 4), fill_value=255, dtype=np.uint8), mode='RGBA'))
    # mesh.visual.material.baseColorTexture = mesh.visual.material.baseColorTexture.resize((2048, 2048))
    texture = Texture.from_trimesh(mesh).to(device='cuda')
    texture.reset_map_Kd_mask()
    mesh = texture.mesh
    map_Kd = texture.map_Kd
    v_rgb = texture.v_rgb

    # refine texture
    intrinsics = dataset_dict['intrinsic']
    c2ws = dataset_dict['frames_c2w']
    images = dataset_dict['frames_img']

    # print("2 vertices:", mesh.v_pos.shape)
    # print("2 faces:", mesh.t_pos_idx.shape)

    # stage 1: initialization
    v_rgb = remapping_vertex_color(
        mesh, c2ws, intrinsics, images, v_rgb,
        proj=projections,
        weights=[1.0] * len(weights),
        confidence=0.2,
        use_alpha=True,
        use_inpainting=True,
        visualize=visualize,
        render_size=render_size,
        b_perspective=b_perspective,
        use_soft_overlay=False
    )
    map_Kd = initial_map_Kd_with_v_rgb(
        mesh, v_rgb,
        texture_size=2048,
        visualize=visualize,
    )

    save_trimesh(mesh.to_trimesh(), (v_rgb*255).to(torch.uint8), "debug/color_project.obj")
    save_image(map_Kd[None,...].permute(0,3,1,2), "debug/map_Kd.png")
    
    # stage 2: refine from 4 views
    map_Kd = remapping_uv_texture(
        mesh, c2ws[refine_index], intrinsics[refine_index], images[refine_index], map_Kd,
        proj=projections,
        weights=weights[refine_index],
        use_alpha=True,
        use_inpainting=False, use_dilation=True,
        texture_size=2048, confidence=0.8,
        visualize=visualize,
        render_size=render_size,
        b_perspective=b_perspective
    )
    textured_mesh = Texture(mesh, v_rgb=None, map_Kd=map_Kd)

    return textured_mesh

