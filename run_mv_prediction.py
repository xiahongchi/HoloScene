import os
from typing import Dict, Optional, Tuple, List
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass

from collections import defaultdict

import torch

import torch.utils.checkpoint
from mv_diffusion_30.models.unet_mv2d_condition import UNetMV2DConditionModel

from mv_diffusion_30.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset

from mv_diffusion_30.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

from einops import rearrange
import rembg
from torchvision.utils import make_grid, save_image

import torchvision.transforms as transforms
from scipy.ndimage import label, sum as ndi_sum

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io


def remove_bg_with_rembg_sam(image, predictor):
    """
    Remove background using rembg initial mask and SAM refinement

    Args:
        image: numpy array with range (0, 1) np.float32 and shape (h, w, 3)

    Returns:
        mask: binary mask where 1 indicates foreground, 0 indicates background
    """
    # Convert float32 (0-1) to uint8 (0-255) for rembg
    print("image: ", image.shape, image.min(), image.max())
    img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

    # # Convert to PIL Image for rembg
    # pil_image = Image.fromarray(img_uint8)
    #
    # # Use rembg to get initial mask
    # # Remove background and get output with alpha channel
    #
    # rembg_session = rembg.new_session()
    # rembg_output = rembg.remove(pil_image, alpha_matting=True, session=rembg_session)
    #
    # # Extract alpha channel (mask)
    # rembg_mask = np.array(rembg_output)[:, :, 3] > 0
    #
    # # Find bounding box of the mask
    # y_indices, x_indices = np.where(rembg_mask)
    # if len(y_indices) == 0 or len(x_indices) == 0:
    #     # If rembg didn't find anything, return the rembg mask
    #     return rembg_mask.astype(np.float32)
    #
    # # Get bounding box with some padding
    # x_min, x_max = np.min(x_indices), np.max(x_indices)
    # y_min, y_max = np.min(y_indices), np.max(y_indices)


    # Add padding (10% of width/height)
    h, w = img_uint8.shape[:2]

    x_min = 0.15 * w
    x_max = 0.85 * w
    y_min = 0.15 * h
    y_max = 0.85 * h

    x_pad = int(0.1 * (x_max - x_min))
    y_pad = int(0.1 * (y_max - y_min))

    # Ensure bounds are within image
    x_min = max(0, x_min - x_pad)
    y_min = max(0, y_min - y_pad)
    x_max = min(w - 1, x_max + x_pad)
    y_max = min(h - 1, y_max + y_pad)

    # Create bbox in format expected by SAM [x, y, x, y]
    bbox = np.array([x_min, y_min, x_max, y_max])
    print("bbox: ", bbox)

    # Set image in predictor
    predictor.set_image(img_uint8)

    # Get SAM prediction using the bounding box
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],  # Add batch dimension
        multimask_output=False  # Get single best mask
    )

    # Take the first mask (best one)
    sam_mask = masks[0]

    # In some cases, SAM might fail to segment properly
    # Fall back to rembg mask if SAM mask looks problematic
    # sam_mask_area = sam_mask.sum()
    # rembg_mask_area = rembg_mask.sum()

    # # If SAM mask is too small or too large compared to rembg mask
    # if sam_mask_area < 0.5 * rembg_mask_area or sam_mask_area > 2.0 * rembg_mask_area:
    #     return rembg_mask.astype(np.float32)

    return sam_mask.astype(np.float32)


def visualize_normal_direction_with_arrow(normal_map, step=10, length=0.05):
    """
    Visualizes a normal map using 3D arrows and returns the visualization as a PIL image.

    Parameters:
    -----------
    normal_map : np.ndarray of shape [H, W, 3]
        The normal map with values in range (-1, 1).
    step : int
        Sampling step to skip pixels for clarity.
    length : float
        Length of the quiver arrows.

    Returns:
    --------
    visualization_image_pil : PIL.Image
        The visualization image in PIL format.
    """
    H, W, _ = normal_map.shape
    # normal_map = normal_map * 2.0 - 1.0

    # Create a grid of pixel coordinates
    y_vals, x_vals = np.mgrid[0:H, 0:W]

    # Sample the grid to reduce clutter
    x_vals = x_vals[::step, ::step]
    y_vals = y_vals[::step, ::step]

    # Flatten after subsampling
    x_vals_f = x_vals.flatten()
    y_vals_f = y_vals.flatten()

    # Extract the corresponding normals
    normals_sampled = normal_map[::step, ::step, :].reshape(-1, 3)

    # Normal vectors
    nx = normals_sampled[:, 0] * 10.0
    ny = normals_sampled[:, 1] * 10.0
    nz = normals_sampled[:, 2]

    # Z-coordinates of arrows' base
    z_vals_f = np.zeros_like(x_vals_f)

    # Create a 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot quivers
    ax.quiver(x_vals_f, y_vals_f, z_vals_f,
              nx, ny, nz,
              length=length,
              normalize=True,
              color='red')

    # Set labels and limits
    ax.set_xlabel('X-axis (image column)')
    ax.set_ylabel('Y-axis (image row)')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Normal Visualization')

    # Adjust viewing angle
    ax.view_init(elev=30, azim=-60)

    # Save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the plot to free memory

    # Convert to PIL image
    buf.seek(0)
    visualization_image_pil = Image.open(buf)

    return visualization_image_pil


import numpy as np
import cv2


def draw_normal_arrows_on_image(
        normal_map,
        step=10,
        scale=5.0,
        color=(255, 255, 255),
        thickness=1,
        tipLength=0.3
):
    """
    Draws 2D arrows representing the (x, y) direction of each pixel’s 3D normal
    directly on the original image.

    Parameters:
    -----------
    image : np.ndarray of shape [H, W, 3]
        The original image (assumed BGR if you’re using standard OpenCV).
    normal_map : np.ndarray of shape [H, W, 3]
        The normal map with (n_x, n_y, n_z) in camera coordinates (range -1 to 1).
    step : int
        Sampling step (larger step = fewer arrows, less clutter).
    scale : float
        A multiplier for how long each arrow should be in the image plane.
    color : tuple of (B, G, R)
        The color for the arrows (default red in OpenCV’s BGR).
    thickness : int
        Line thickness for arrows.
    tipLength : float
        Size of the arrow tip relative to the arrow length (OpenCV’s arrowedLine param).

    Returns:
    --------
    output_image : np.ndarray of shape [H, W, 3]
        A copy of the original image with arrows drawn on top.
    """
    # Make a copy so we don’t modify the original
    output_image = (normal_map.copy() * 0.5 + 0.5) * 255.0
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    H, W, _ = normal_map.shape

    # Iterate over subsampled grid
    for y in range(0, H, step):
        for x in range(0, W, step):
            # Normal vector at (y, x)
            nx, ny, nz = normal_map[y, x]

            # 2D arrow end point
            end_x = int(x + scale * nx)
            end_y = int(y + scale * ny)

            # Draw the arrow on the image
            cv2.arrowedLine(
                output_image,
                (x, y),
                (end_x, end_y),
                color=color,
                thickness=thickness,
                tipLength=tipLength
            )
    output_image = Image.fromarray(output_image)
    return output_image


weight_dtype = torch.half

VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

to_pil = transforms.ToPILImage()
@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: Optional[str]
    revision: Optional[str]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation
    load_task: bool

def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # pdb.set_trace()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_depth_numpy(depth, fp, alpha):
    depth = depth.mul(0.4).mul(65535.).add_(0.5).to("cpu", torch.float32).numpy().mean(0)
    print(depth.min(), depth.max())

    depth[alpha < 128] = 0

    depth = depth.astype(np.uint16)

    kernel = np.ones((3, 3), np.uint8)  # kernel for erode

    # erode
    depth = cv2.erode(depth, kernel, iterations=1)

    im = Image.fromarray(depth)
    im.save(fp)


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def load_wonder3d_pipeline(cfg):
    if cfg.pretrained_unet_path:
        print("load pre-trained unet from ", cfg.pretrained_unet_path)
        unet = UNetMV2DConditionModel.from_pretrained(cfg.pretrained_unet_path, revision=cfg.revision,
                                                               **cfg.unet_from_pretrained_kwargs)

    pipeline = MVDiffusionImagePipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        pred_type=cfg.pred_type,
        safety_checker=None,
        unet=unet
    )

    if torch.cuda.is_available():
        pipeline.to('cuda:0')

    return pipeline


def largest_connected_region(binary_image):
    # Label connected components
    labeled_array, num_features = label(binary_image)

    if num_features == 0:
        return np.zeros_like(binary_image, dtype=bool)  # No connected region found

    # Compute size of each component
    component_sizes = ndi_sum(binary_image, labeled_array, index=range(1, num_features + 1))

    # Find the largest component label
    largest_label = np.argmax(component_sizes) + 1  # +1 because labels start from 1

    # Create a mask for the largest component
    largest_component_mask = labeled_array == largest_label

    return largest_component_mask

def views_6to3(imgs):
    outs = []
    for i in range(6):
        if i == 0 or i == 1 or i == 5:
            continue
        outs.append(imgs[i])
    return outs

def views_6to4(imgs):
    outs = []
    for i in range(6):
        if i == 1 or i == 5:
            continue
        outs.append(imgs[i])
    return outs

def pred_multiview_joint(image, pipeline, seed=42, crop_size=192, camera_type='ortho', cfg=None, case_name='img', output_path='outputs'):

    validation_dataset = MVDiffusionDataset(
        single_image=image,
        num_views=6,
        bg_color='white',
        img_wh=[256, 256],
        crop_size=crop_size,
        cam_types=[camera_type],
        load_cam_type=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)

    batch = next(iter(validation_dataloader))

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0)

    filename = batch['filename']

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
    # (B*Nv, Nce)
    camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce").to(weight_dtype)

    images_cond.append(imgs_in)
    num_views = len(VIEWS)
    with torch.autocast("cuda"):
        # B*Nv images
        for guidance_scale in cfg.validation_guidance_scales:
            out = pipeline(
                imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
                output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
            ).images

            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:]
            color_pred_grid = make_grid(images_pred, nrow=6, padding=0, value_range=(0, 1))
            normal_pred_grid = make_grid(normals_pred, nrow=6, padding=0, value_range=(0, 1))

            rm_normals = []
            colors = []
            for i in range(bsz // num_views):
                # scene = os.path.basename(case_name.split('.')[0])
                scene_dir = output_path

                # normal_dir = os.path.join(scene_dir, "normals")
                # color_dir = os.path.join(scene_dir, "colors")
                # masked_colors_dir = os.path.join(scene_dir, "masked_colors")
                # os.makedirs(normal_dir, exist_ok=True)
                # os.makedirs(masked_colors_dir, exist_ok=True)
                # os.makedirs(color_dir, exist_ok=True)
                # print(scene, batch['cam_type'], scene_dir)
                rembg_session = rembg.new_session()
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i * num_views + j
                    normal = normals_pred[idx]
                    color = images_pred[idx]

                    normal_filename = f"normals_{view}.png"
                    rgb_filename = f"rgb_{view}.png"
                    rm_normal_filename = f"rm_normals_{view}.png"
                    normal = save_image(normal, os.path.join(scene_dir, normal_filename))
                    color = save_image(color, os.path.join(scene_dir, rgb_filename))

                    rm_normal = rembg.remove(normal, alpha_matting=True, session=rembg_session)
                    # rm_normal_alpha = rm_normal[:, :, 3] > 0
                    # rm_normal_alpha = largest_connected_region(rm_normal_alpha)
                    # rm_normal = np.concatenate([
                    #     rm_normal[..., :3],
                    #     np.clip(rm_normal_alpha.astype(np.float32) * 255, 0, 255).astype(np.uint8)[..., None]
                    #     ], axis=-1)
                    rm_normal = rm_normal.copy()
                    # rm_normal[..., 1:3] = 255 - rm_normal[..., 1:3]
                    rm_normal = Image.fromarray(rm_normal)
                    rm_normal.save(os.path.join(scene_dir, rm_normal_filename))
                    rm_normals.append(rm_normal)
                    colors.append(to_pil(color))

                    # save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))

            save_image(color_pred_grid, os.path.join(scene_dir, f'color_grid_img.png'))
            save_image(normal_pred_grid, os.path.join(scene_dir, f'normal_grid_img.png'))

            normals, colors = views_6to3(rm_normals), views_6to3(colors)
            angles = [-90, 180, 90]
            angles_rad = [angle * np.pi / 180.0 for angle in angles]
            for angle_i, angle in enumerate(angles_rad):
                rot = np.eye(3)
                rot[2, 2] = np.cos(angle)
                rot[2, 0] = -np.sin(angle)
                rot[0, 2] = np.sin(angle)
                rot[0, 0] = np.cos(angle)

                normal = normals[angle_i]
                normal = (np.array(normal) / 255.) * 2 - 1
                normal[..., :3] = np.dot(normal[..., :3], rot.T)

                normal[..., 1:3] = -normal[..., 1:3]

                normal_vis = draw_normal_arrows_on_image(normal[..., :3] * normal[..., 3:])

                normal = (normal + 1) / 2 * 255
                normal = Image.fromarray(normal.astype(np.uint8))
                normals[angle_i] = normal

                rm_normal_filename = f"normals_cam_angle_{angle_i}.png"
                normal.save(os.path.join(scene_dir, rm_normal_filename))
                normal_vis.save(os.path.join(scene_dir, f'normal_vis_cam_angle_{angle_i}.png'))
            return normals, colors



def pred_multiview_joint_simple(image, pipeline, seed=42, crop_size=192, camera_type='ortho', cfg=None, case_name='img'):

    validation_dataset = MVDiffusionDataset(
        single_image=image,
        num_views=6,
        bg_color='white',
        img_wh=[256, 256],
        crop_size=crop_size,
        cam_types=[camera_type],
        load_cam_type=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)

    batch = next(iter(validation_dataloader))

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0)

    filename = batch['filename']

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
    # (B*Nv, Nce)
    camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce").to(weight_dtype)

    images_cond.append(imgs_in)
    num_views = len(VIEWS)
    with torch.autocast("cuda"):
        # B*Nv images
        for guidance_scale in cfg.validation_guidance_scales:
            out = pipeline(
                imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
                output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
            ).images

            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:]

            rm_normals = []
            colors = []
            for i in range(bsz // num_views):
                rembg_session = rembg.new_session()
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i * num_views + j
                    normal = normals_pred[idx]
                    color = images_pred[idx]

                    normal = Image.fromarray(np.clip(normal.permute(1, 2, 0).detach().cpu().numpy() * 255., 0, 255).astype(np.uint8))
                    color = color.permute(1, 2, 0)[..., :3].detach().cpu().numpy()

                    rm_normal = rembg.remove(normal, alpha_matting=True, session=rembg_session)
                    rm_normals.append(rm_normal)
                    colors.append(color)


            # normals, colors = views_6to4(rm_normals), views_6to4(colors)
            normals, colors = rm_normals, colors
            alphas = []
            angles = [0, -45, -90, 180, 90, 45]
            angles_rad = [angle * np.pi / 180.0 for angle in angles]
            for angle_i, angle in enumerate(angles_rad):
                rot = np.eye(3)
                rot[2, 2] = np.cos(angle)
                rot[2, 0] = -np.sin(angle)
                rot[0, 2] = np.sin(angle)
                rot[0, 0] = np.cos(angle)

                normal = normals[angle_i]
                normal = (np.array(normal) / 255.) * 2 - 1
                normal[..., :3] = np.dot(normal[..., :3], rot.T)

                normal[..., 1:3] = -normal[..., 1:3]

                # normal = (normal + 1) / 2
                alpha = normal[..., 3] > 0.5
                normals[angle_i] = normal[..., :3]
                alphas.append(alpha)

            return normals, colors, alphas


def pred_multiview_joint_simple_color_rembg(image, pipeline, seed=42, crop_size=192, camera_type='ortho', cfg=None, case_name='img'):

    validation_dataset = MVDiffusionDataset(
        single_image=image,
        num_views=6,
        bg_color='white',
        img_wh=[256, 256],
        crop_size=crop_size,
        cam_types=[camera_type],
        load_cam_type=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)

    batch = next(iter(validation_dataloader))

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0)

    filename = batch['filename']

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
    # (B*Nv, Nce)
    camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce").to(weight_dtype)

    images_cond.append(imgs_in)
    num_views = len(VIEWS)
    with torch.autocast("cuda"):
        # B*Nv images
        for guidance_scale in cfg.validation_guidance_scales:
            out = pipeline(
                imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
                output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
            ).images

            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:]

            normals = []
            colors = []
            alphas = []
            for i in range(bsz // num_views):
                rembg_session = rembg.new_session()
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i * num_views + j
                    normal = normals_pred[idx]
                    color = images_pred[idx]

                    normal = normal.permute(1, 2, 0)[..., :3].detach().cpu().numpy()
                    color = Image.fromarray(np.clip(color.permute(1, 2, 0)[..., :3].detach().cpu().numpy() * 255., 0, 255).astype(np.uint8))

                    color = rembg.remove(color, alpha_matting=True, session=rembg_session)
                    color = np.array(color) / 255.
                    alpha = color[..., 3] > 0.5
                    color = color[..., :3]

                    normals.append(normal)
                    colors.append(color)
                    alphas.append(alpha)


            normals, colors = views_6to4(normals), views_6to4(colors)
            angles = [0, -90, 180, 90]
            angles_rad = [angle * np.pi / 180.0 for angle in angles]
            for angle_i, angle in enumerate(angles_rad):
                rot = np.eye(3)
                rot[2, 2] = np.cos(angle)
                rot[2, 0] = -np.sin(angle)
                rot[0, 2] = np.sin(angle)
                rot[0, 0] = np.cos(angle)

                normal = normals[angle_i]
                normal = normal * 2 - 1
                normal[..., :3] = np.dot(normal[..., :3], rot.T)

                normal[..., 1:3] = -normal[..., 1:3]

                normals[angle_i] = normal[..., :3]

            return normals, colors, alphas

def wonder3d_generation(image_np, mv_pipeline, config_mv, upsampler, upsampler_func, seed=42):
    image_pil = Image.fromarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8))
    front_img = upsampler_func(image_pil, upsampler)

    normals, colors, alphas = pred_multiview_joint_simple(front_img, mv_pipeline, cfg=config_mv, seed=seed)
    return normals, colors, alphas

def pred_multiview_joint_simple_sam(image, pipeline, seed=42, crop_size=192, camera_type='ortho', cfg=None, case_name='img', predictor=None):

    assert predictor is not None, "predictor is not provided"

    validation_dataset = MVDiffusionDataset(
        single_image=image,
        num_views=6,
        bg_color='white',
        img_wh=[256, 256],
        crop_size=crop_size,
        cam_types=[camera_type],
        load_cam_type=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    images_cond, normals_pred, images_pred = [], defaultdict(list), defaultdict(list)

    batch = next(iter(validation_dataloader))

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0)

    filename = batch['filename']

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W").to(weight_dtype)
    # (B*Nv, Nce)
    camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce").to(weight_dtype)

    images_cond.append(imgs_in)
    num_views = len(VIEWS)
    with torch.autocast("cuda"):
        # B*Nv images
        for guidance_scale in cfg.validation_guidance_scales:
            out = pipeline(
                imgs_in, camera_embeddings, generator=generator, guidance_scale=guidance_scale,
                output_type='pt', num_images_per_prompt=1, **cfg.pipe_validation_kwargs
            ).images

            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:]

            rm_normals = []
            colors = []
            alphas = []
            for i in range(bsz // num_views):
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i * num_views + j
                    normal = normals_pred[idx]
                    color = images_pred[idx]

                    normal = normal.permute(1, 2, 0).detach().cpu().numpy()
                    color = color.permute(1, 2, 0)[..., :3].detach().cpu().numpy()

                    # rm_normal = rembg.remove(normal, alpha_matting=True, session=rembg_session)
                    mask = remove_bg_with_rembg_sam(color, predictor)
                    alphas.append(mask)
                    rm_normals.append(normal)
                    colors.append(color)


            normals, colors = views_6to4(rm_normals), views_6to4(colors)
            angles = [0, -90, 180, 90]
            angles_rad = [angle * np.pi / 180.0 for angle in angles]
            for angle_i, angle in enumerate(angles_rad):
                rot = np.eye(3)
                rot[2, 2] = np.cos(angle)
                rot[2, 0] = -np.sin(angle)
                rot[0, 2] = np.sin(angle)
                rot[0, 0] = np.cos(angle)

                normal = normals[angle_i]
                normal = normal * 2 - 1
                normal[..., :3] = np.dot(normal[..., :3], rot.T)

                normal[..., 1:3] = -normal[..., 1:3]

                # normal = (normal + 1) / 2
                normals[angle_i] = normal[..., :3]

            return normals, colors, alphas

def wonder3d_generation_sam(image_np, mv_pipeline, config_mv, upsampler, upsampler_func, predictor):
    image_pil = Image.fromarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8))
    front_img = upsampler_func(image_pil, upsampler)

    normals, colors, alphas = pred_multiview_joint_simple_sam(front_img, mv_pipeline, cfg=config_mv, predictor=predictor)
    return normals, colors, alphas
