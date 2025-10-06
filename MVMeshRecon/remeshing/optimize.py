from tqdm import tqdm
from MVMeshRecon.MeshRecon.opt import MeshOptimizer
from MVMeshRecon.remeshing.core.remesh import calc_vertex_normals
from MVMeshRecon.utils.loss_utils import NormalLoss
import torchvision.utils as vutils
import numpy as np
from PIL import Image

def save_tensor_mask(mask, filename):
    # Squeeze to remove any extra dimensions
    mask = mask.squeeze(0)

    # Ensure the mask is in a 0-1 range and in the right shape (CxHxW)
    vutils.save_image(mask, filename)

def do_optimize(vertices, faces, ref_images, renderer, weights, remeshing_steps, edge_len_lims=(0.01, 0.1), decay=0.999, debug_dir=None):
    # optimizer initialization
    opt = MeshOptimizer(vertices, faces, local_edgelen=False, edge_len_lims=edge_len_lims, gain=0.1)
    vertices = opt.vertices

    # normal optimization step
    loss_func = NormalLoss(mask_loss_weights = 10.)
    for i in tqdm(range(remeshing_steps)):
        opt.zero_grad()
        opt._lr *= decay

        normals = calc_vertex_normals(vertices, faces)
        render_normal = renderer.render_normal(vertices, normals, faces)

        # image_label = f"iter_{i:03d}"
        # for image_i in range(ref_images.shape[0]):
        #     Image.fromarray(
        #         np.clip(render_normal[image_i].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(
        #             f"{debug_dir}/{image_label}_{image_i:0>2d}_render.png")
        #     Image.fromarray(
        #         np.clip(ref_images[image_i].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(
        #         f"{debug_dir}/{image_label}_{image_i:0>2d}_ref.png")

        loss_expand = 0.5 * ((vertices + normals).detach() - vertices).pow(2).mean()

        # Extract mask and ground truth mask
        mask = render_normal[..., [3]]
        gtmask = ref_images[..., [3]]

        # Compute loss with the mask
        loss = loss_func(render_normal, ref_images, weights=weights, mask=mask, gtmask=gtmask)
        loss_expansion_weight = 0.1
        loss = loss + loss_expansion_weight * loss_expand

        loss.backward()
        opt.step()
        vertices, faces = opt.remesh(poisson=False)

    return vertices.detach(), faces.detach()