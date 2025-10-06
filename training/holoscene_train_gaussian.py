import imp
import os
import shutil
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import json
import wandb
from torch.utils.tensorboard import SummaryWriter
import trimesh
import pickle
import cv2

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import (
    get_time, refined_obj_bbox, get_lcc_mesh, get_scale_shift, apply_scale_shift,
    sample_views_around_object, apply_inv_scale_shift, build_camera_matrix, get_camera_orthogonal_rays,
    vis_prune, sample_views_around_object_backface, sample_views_around_object_naive, visualize_view_weights,
    get_camera_orthogonal_projection_matrix, rasterize_mesh_list, get_cam_normal_from_rast, get_faces_normal, get_camera_perspective_rays,
    find_best_additional_view, evaluate_view_addition, margin_aware_fps_sampling, highest_sampling_view_weights,
    find_largest_connected_region, build_camera_matrix_from_angles_and_locs, rasterize_mesh_list_front_face,
    occlusion_test, cluster_points, fov_to_focal_length, get_camera_perspective_projection_matrix,
    rasterize_trimesh, get_camera_perspective_rays_world, find_longest_continuous_azimuths, get_fg_mask_rembg,
    build_camera_matrix_from_angles_and_locs_diff, get_theta_phi, align_normal_pred_lama_omnidata,
    visualize_view_weights_with_highlighted_azimuths, find_diff_color, resize_int_tensor,
    total_variation_loss, second_order_smoothness, get_normal_map_from_depth, rasterize_mesh, smooth_rgb_image, get_fg_occulusion_mask,
    generate_traverse_seq, rasterize_mesh_return_pixel_vert_and_bary, get_camera_orthogonal_projection_matrix_offset,
    load_tex_dict_from_tex_mesh_p3d, get_tex_mesh_dict_for_nvrast, rasterize_mesh_with_uv
)
from utils.plots import (
    remesh, simplify_mesh
)
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
from PIL import Image
import torch.nn.functional as F
import nvdiffrast.torch as dr
from omegaconf import OmegaConf
from run_mv_prediction import load_wonder3d_pipeline, wonder3d_generation, wonder3d_generation_sam
from upsample.rrdbnet import RRDBNet
from upsample.upsampler import RealESRGANer
from upsample.refine_lr_to_sr import sr_front_with_upsampler
from lama.utils import load_model, inpaint
from scipy.ndimage import binary_dilation, binary_erosion
from midas.omnidata import load_normal_model, infer_normal
from segment_anything import sam_model_registry, SamPredictor
import subprocess
import xatlas
# from pytorch3d.io import save_obj
from pytorch3d.io import load_objs_as_meshes, save_obj

from sklearn.neighbors import NearestNeighbors
import torchvision
from model.network import ColorImplicitNetworkSingle
from model.gom import GoM, SplatfactoOnMeshUCModelConfig
from model.gs import GS
from collections import OrderedDict
from plyfile import PlyData


def compute_scale_and_shift_batch(prediction, target):
    # prediction: (B, N), N = 32
    # target: (B, N)
    B, N = prediction.shape
    dr = prediction.unsqueeze(-1) # (B, N, 1)
    dr = torch.cat((dr, torch.ones_like(dr).to(dr.device)), dim=-1).reshape(-1, 2, 1)  # (BxN, 2, 1)
    dr_sq = torch.sum((dr @ dr.transpose(1, 2)).reshape(B, N, 2, 2), dim=1) # (B, 2, 2)
    left_part = torch.inverse(dr_sq).reshape(B, 2, 2) # (B, 2, 2)
    right_part = torch.sum((dr.reshape(B, N, 2, 1))*(target.reshape(B, N, 1, 1)), dim=1).reshape(B, 2, 1)
    rs = left_part @ right_part # (B, 2, 1)
    rs = rs.reshape(B, 2)
    return rs[:, 0], rs[:, 1]

def depth_prior_loss(depth_pred, depth_gt):
    depth_gt = depth_gt.reshape(1, -1)  # (B, N)
    depth_pred = depth_pred.reshape(1, -1)  # (B, N)
    w, q = compute_scale_and_shift_batch(depth_pred, depth_gt)

    w = w.reshape(-1, 1)
    q = q.reshape(-1, 1)

    # w: (B, 1); q: (B, 1)
    diff = ((w * depth_pred + q) - depth_gt) ** 2
    diff = torch.clip(diff, max=1)
    return diff.mean()

def read_ply(filename):
    """
    Reads a PLY file and extracts the vertex properties.

    Parameters:
    filename (str): The name of the file to read.

    Returns:
    OrderedDict: Dictionary mapping property names to numpy arrays
    int: Number of vertices
    """
    plydata = PlyData.read(filename)
    vertices = plydata['vertex']

    # Get count
    count = len(vertices)

    # Create ordered dictionary to store properties
    properties = OrderedDict()

    # Extract all properties
    for prop in vertices.properties:
        name = prop.name
        properties[name] = np.array(vertices[name])

    return properties, count


def load_gaussian_from_ply(filename, device="cuda", dtype=torch.float32):
    """
    Loads Gaussian parameters from a PLY file.

    Parameters:
    filename (str): The name of the PLY file.
    device (str or torch.device): Device to put tensors on.
    dtype (torch.dtype): Data type for tensors.

    Returns:
    dict: Dictionary containing the model parameters:
        - means: Positions of Gaussians (N, 3)
        - opacities: Opacity values (N,)
        - features_dc: SH DC components (N, 3, 1) or None if using direct colors
        - features_rest: SH rest components (N, 3, C) or None if using direct colors
        - colors: Direct colors (N, 3) or None if using SH
        - scales: Scale values (N, 3)
        - quats: Quaternion values (N, 4)
    """
    # Read PLY file
    properties, count = read_ply(filename)

    # Extract positions (means)
    means = np.stack([properties['x'], properties['y'], properties['z']], axis=1)
    means_tensor = torch.tensor(means, dtype=dtype, device=device)

    # Extract opacities
    opacities = properties['opacity']
    opacities_tensor = torch.tensor(opacities, dtype=dtype, device=device)

    # Check if using SH or direct colors
    features_dc = None
    features_rest = None
    colors = None

    # Check if we have SH components
    dc_keys = [k for k in properties.keys() if k.startswith('f_dc_')]
    rest_keys = [k for k in properties.keys() if k.startswith('f_rest_')]
    assert rest_keys

    if dc_keys and rest_keys:
        # Using spherical harmonics
        # Extract features_dc (SH DC components)
        dc_dim = len(dc_keys)
        features_dc_np = np.zeros((count, dc_dim, 1), dtype=np.float32)
        for i, key in enumerate(dc_keys):
            features_dc_np[:, i, 0] = properties[key]
        features_dc = torch.tensor(features_dc_np, dtype=dtype, device=device)

        # Extract features_rest (SH rest components)
        rest_dim = len(rest_keys)
        features_rest_np = np.zeros((count, rest_dim), dtype=np.float32)
        for i, key in enumerate(rest_keys):
            features_rest_np[:, i] = properties[key]

        # Calculate the SH degree based on the number of components
        # Number of SH coefficients = (degree+1)²
        # So rest components = (degree+1)² - 1
        sh_degree = int(np.sqrt(rest_dim / 3 + 1) - 1)

        # Reshape to match expected dimensions (N, 3, C)
        # Where C is (degree²+2*degree)
        features_rest_np = features_rest_np.reshape(count, (sh_degree * sh_degree + 2 * sh_degree), 3)
        # features_rest_np = np.transpose(features_rest_np, (0, 2, 1))
        features_rest = torch.tensor(features_rest_np, dtype=dtype, device=device)
    elif 'colors' in properties:
        # Using direct colors
        colors_np = properties['colors'].astype(np.float32) / 255.0
        colors = torch.tensor(colors_np, dtype=dtype, device=device)

    # Extract scales
    scale_keys = [k for k in properties.keys() if k.startswith('scale_')]
    scales_np = np.zeros((count, len(scale_keys)), dtype=np.float32)
    for i, key in enumerate(scale_keys):
        scales_np[:, i] = properties[key]
    scales = torch.tensor(scales_np, dtype=dtype, device=device)

    # Extract quaternions
    rot_keys = [k for k in properties.keys() if k.startswith('rot_')]
    quats_np = np.zeros((count, len(rot_keys)), dtype=np.float32)
    for i, key in enumerate(rot_keys):
        quats_np[:, i] = properties[key]
    quats = torch.tensor(quats_np, dtype=dtype, device=device)

    # Return all components
    return {
        'means': means_tensor,
        'opacities': opacities_tensor,
        'features_dc': features_dc.reshape(-1, 3),
        'features_rest': features_rest,
        'scales': scales,
        'quats': quats,
        "sh_degree": sh_degree
    }

class HoloSceneTrainGaussianRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.description = kwargs['description']
        self.use_wandb = False # kwargs['use_wandb']
        self.ft_folder = kwargs['ft_folder']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        self.scan_id = scan_id
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        self.finetune_folder = kwargs['ft_folder'] if kwargs['ft_folder'] is not None else None
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('./', kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join('./', kwargs['exps_folder_name'], self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('./', self.exps_folder_name))
        self.expdir = os.path.join('./', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        if self.description == "":              # default not use description
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        else:
            self.timestamp = f'{self.description}' + '_{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        print('[INFO]: shell command : {0}'.format(' '.join(sys.argv)))

        print('[INFO]: Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        self.invis_loss_conf = self.conf.get_config('invis_loss')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.ds_len = len(self.train_dataset)
        self.n_sem = len(self.train_dataset.label_mapping)
        self.max_total_iters = self.n_sem * 200
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))


        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=1,
                                                            shuffle=True,
                                                            num_workers=8,
                                                            pin_memory=True)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           )

        self.start_epoch = int(kwargs['checkpoint'])
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')
            self.finetune_folder = os.path.join(self.expdir, timestamp)

            # continue training need copy mesh files from old folder
            old_plots_folder = os.path.join(self.finetune_folder, 'plots')

            num_objs = self.n_sem
            for obj_i in range(num_objs):
                mesh_obj_i_name = f'coarse_recon_obj_{obj_i}.ply'
                mesh_obj_i_path = os.path.abspath(os.path.join(old_plots_folder, mesh_obj_i_name))
                new_ln_path = os.path.join(self.plots_dir, mesh_obj_i_name)
                if os.path.exists(mesh_obj_i_path):
                    os.symlink(mesh_obj_i_path, new_ln_path)

                mesh_obj_i_name = f'surface_{obj_i}.obj'
                mesh_obj_i_path = os.path.abspath(os.path.join(old_plots_folder, mesh_obj_i_name))
                new_ln_path = os.path.join(self.plots_dir, mesh_obj_i_name)
                if os.path.exists(mesh_obj_i_path):
                    os.symlink(mesh_obj_i_path, new_ln_path)

                mesh_obj_i_name = f'surface_{obj_i}.mtl'
                mesh_obj_i_path = os.path.abspath(os.path.join(old_plots_folder, mesh_obj_i_name))
                new_ln_path = os.path.join(self.plots_dir, mesh_obj_i_name)
                if os.path.exists(mesh_obj_i_path):
                    os.symlink(mesh_obj_i_path, new_ln_path)

                mesh_obj_i_name = f'surface_{obj_i}.png'
                mesh_obj_i_path = os.path.abspath(os.path.join(old_plots_folder, mesh_obj_i_name))
                new_ln_path = os.path.join(self.plots_dir, mesh_obj_i_name)
                if os.path.exists(mesh_obj_i_path):
                    os.symlink(mesh_obj_i_path, new_ln_path)

                mesh_obj_i_name = f"gauss_obj_{obj_i}.pt"
                mesh_obj_i_path = os.path.abspath(os.path.join(old_plots_folder, mesh_obj_i_name))
                if os.path.exists(mesh_obj_i_path):
                    new_ln_path = os.path.join(self.plots_dir, mesh_obj_i_name)
                    os.symlink(mesh_obj_i_path, new_ln_path)

                if obj_i > 0:
                    vis_info_obj_i_name = f'vis_info_{obj_i}.pkl'
                    vis_info_obj_i_path = os.path.abspath(os.path.join(old_plots_folder, vis_info_obj_i_name))
                    new_ln_path = os.path.join(self.plots_dir, vis_info_obj_i_name)
                    if os.path.exists(vis_info_obj_i_path):
                        os.symlink(vis_info_obj_i_path, new_ln_path)
                else:
                    bg_info_name = f'bg_info.pkl'
                    bg_info_path = os.path.abspath(os.path.join(old_plots_folder, bg_info_name))
                    new_ln_path = os.path.join(self.plots_dir, bg_info_name)
                    if os.path.exists(bg_info_path):
                        os.symlink(bg_info_path, new_ln_path)
                
                translation_dict_path = os.path.abspath(os.path.join(old_plots_folder, 'translation_dict.pkl'))
                new_ln_path = os.path.abspath(os.path.join(self.plots_dir, 'translation_dict.pkl'))
                if os.path.exists(translation_dict_path):
                    os.symlink(translation_dict_path, new_ln_path)
                
                graph_node_dict_path = os.path.abspath(os.path.join(old_plots_folder, 'graph_node_dict.pkl'))
                new_ln_path = os.path.abspath(os.path.join(self.plots_dir, 'graph_node_dict.pkl'))
                if os.path.exists(graph_node_dict_path):
                    os.symlink(graph_node_dict_path, new_ln_path)


        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()


        self.glctx = dr.RasterizeGLContext()

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):

        print("training...")

        if self.use_wandb:
            infos = json.loads(json.dumps(self.conf))
            wandb.init(
                config=infos,
                project=self.conf.get_string('wandb.project_name'),
                name=self.timestamp,
            )

        else:
            print('Not using wandb, use tensorboard instead.')
            log_dir = os.path.join(self.expdir, self.timestamp, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir)

        self.iter_step = self.start_epoch * len(self.train_dataset)
        print(f'Start epoch: {self.start_epoch}, iter_step: {self.iter_step}')

        all_meshes = []
        num_objs = self.n_sem
        vis_info_list = []

        for obj_i in range(num_objs):
            obj_i_mesh_path = os.path.join(self.plots_dir, f'coarse_recon_obj_{obj_i}.ply')
            assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
            obj_i_mesh = trimesh.exchange.load.load_mesh(obj_i_mesh_path)
            all_meshes.append(obj_i_mesh)


            if obj_i > 0:
                vis_info_obj_i_name = f'vis_info_{obj_i}.pkl'
                vis_info_obj_i_path = os.path.join(self.plots_dir, vis_info_obj_i_name)
                with open(vis_info_obj_i_path, 'rb') as f:
                    vis_info_obj_i = pickle.load(f)
                vis_info_list.append(vis_info_obj_i)
            else:
                bg_info_name = f'bg_info.pkl'
                bg_info_path = os.path.join(self.plots_dir, bg_info_name)
                with open(bg_info_path, 'rb') as f:
                    bg_info = pickle.load(f)
                vis_info_list.append(bg_info)

        scene_mesh = trimesh.util.concatenate(all_meshes)
        scene_mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(scene_mesh.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(scene_mesh.faces).int().to("cuda").contiguous(),
            'vertices_world': torch.from_numpy(scene_mesh.vertices).float().to("cuda").contiguous(),
        }


        self.model = GoM(
            seed_mesh=[os.path.join(self.plots_dir, f'surface_{obj_i}.obj') for obj_i in range(num_objs)],
            area_to_subdivide=self.conf['model'].get("area_to_subdivide", 1e-5),
            additional_configs=self.conf['model']
        )

        instance_gs_idxs = self.model.instance_gs_idxs
        self.instance_gs_idxs = instance_gs_idxs = [gs_idx.cuda() for gs_idx in instance_gs_idxs]

        print("len(instance_gs_idxs): ", len(instance_gs_idxs))

        if torch.cuda.is_available():
            self.model.cuda()
            self.model.device = torch.device("cuda")

        # The MLP and hash grid should have different learning rates
        self.lr = self.conf.get_float('train.learning_rate')

        model_params = self.model.get_param_groups()

        self.optimizer = torch.optim.Adam([
            {'name': 'means_2d', 'params': model_params['means_2d'],
             'lr': 1.6e-4},
            {'name': 'normal_elevates', 'params': model_params['normal_elevates'],
             'lr': 1.6e-4},
            {'name': 'features_dc', 'params': model_params['features_dc'],
             'lr': 0.0025},
            {'name': 'features_rest', 'params': model_params['features_rest'],
             'lr': 0.0025 / 20},
            {'name': 'opacities', 'params': model_params['opacities'],
             'lr': 0.05},
            {'name': 'scales', 'params': model_params['scales'],
             'lr': 0.005},
            {'name': 'quats', 'params': model_params['quats'],
             'lr': 0.001},
        ], betas=(0.9, 0.99), eps=1e-15)

        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        total_steps = self.max_total_iters
        decay_steps = total_steps
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1. / decay_steps))

        self.iter_step = 0
        val_dir = os.path.join(self.plots_dir, f'val')
        os.makedirs(val_dir, exist_ok=True)

        mesh_obj_dict_list = []

        for obj_i in range(num_objs):

            mesh_obj_i = all_meshes[obj_i]

            mesh_obj_i_dict = {
                'vertices': F.pad(
                    torch.from_numpy(mesh_obj_i.vertices).float().to("cuda").contiguous(),
                    pad=(0, 1), value=1.0, mode='constant'),
                'pos_idx': torch.from_numpy(mesh_obj_i.faces).int().to("cuda").contiguous(),
                'vertices_world': torch.from_numpy(mesh_obj_i.vertices).float().to("cuda").contiguous(),
            }

            mesh_obj_dict_list.append(mesh_obj_i_dict)

        self.train_dataset.sampling_class_id = -1

        for epoch in range(self.nepochs + 1):

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad()

                fx, fy, cx, cy = model_input["intrinsics"]
                pose = model_input["pose"].reshape(4, 4).float().cuda()
                image_res = model_input["image_res"]
                near, far = model_input["near_far"]

                H, W = int(image_res[0]), int(image_res[1])
                fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
                near, far = float(near), float(far)

                cx = cx # + np.random.rand() - 0.5
                cy = cy # + np.random.rand() - 0.5

                rgb_gt = ground_truth["rgb"].cuda().reshape(H, W, 3)
                depth_gt = ground_truth["depth"].cuda().reshape(H, W)
                # instance_mask = ground_truth["mask"].cuda().reshape(H, W).int() == obj_i

                camera_projmat = get_camera_perspective_projection_matrix(fx, fy, cx, cy, H, W, near, far)
                camera_projmat = torch.from_numpy(camera_projmat).reshape(4, 4).float().cuda()

                mvp = camera_projmat @ torch.inverse(pose)

                H_sr = H
                W_sr = W
                while W_sr % 8 != 0 or H_sr % 8 != 0:
                    W_sr *= 2
                    H_sr *= 2

                valid, _, mesh_depth = rasterize_mesh(scene_mesh_dict, mvp, self.glctx, (H_sr, W_sr), pose)

                if H_sr != H or W_sr != W:
                    valid = torchvision.transforms.functional.resize(valid.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)

                    mesh_depth = mesh_depth.unsqueeze(-1)

                    mesh_depth = mesh_depth.permute(2, 0, 1)
                    mesh_depth = torchvision.transforms.functional.resize(mesh_depth, (H, W), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                    mesh_depth = mesh_depth.permute(1, 2, 0).squeeze(-1)

                K = torch.from_numpy(
                    np.array([
                        fx, 0, cx,
                        0, fy, cy,
                        0, 0, 1
                    ], dtype=np.float32).reshape(3, 3)
                ).cuda().float()

                model_outputs = self.model.get_outputs(
                    pose=pose,
                    K=K,
                    H=H,
                    W=W,
                    camera_model="pinhole"
                )

                rgb = model_outputs['rgb'].reshape(H, W, 3)
                depth = model_outputs['depth'].reshape(H, W, 1)
                accumulation = model_outputs['accumulation'].reshape(H, W, 1)

                batch_gs = {
                    "acm": valid.reshape(H, W, 1).float(),
                    "mesh_depth": mesh_depth.reshape(H, W, 1).float(),
                    "image": torch.cat([rgb_gt.reshape(H, W, 3), valid.reshape(H, W, 1).float()], dim=-1),
                }

                model_outputs = {
                    'rgb': rgb,
                    'depth': depth,
                    'accumulation': accumulation,
                    'background': model_outputs['background'],
                }

                main_loss_dict = self.model.get_loss_dict(model_outputs, batch_gs)

                loss = main_loss_dict["main_loss"] * 5.0 + main_loss_dict["scale_reg"]

                if self.conf['train'].get('lambda_depth_prior', 0.) > 0.:
                    depth_loss = depth_prior_loss(depth, depth_gt)
                    loss += depth_loss * self.conf['train'].get('lambda_depth_prior', 0.)
                    print("depth_loss: ", depth_loss.item())

                if self.iter_step % 200 == 0:
                    with torch.no_grad():
                        Image.fromarray(
                            np.clip(rgb_gt.detach().cpu().numpy().reshape(H, W, 3) * 255., 0, 255).astype(
                                np.uint8)).save(os.path.join(val_dir, f'{self.iter_step}_gt.png'))
                        Image.fromarray(
                            valid.detach().cpu().numpy().reshape(H, W).astype(np.uint8) * 255).save(
                            os.path.join(val_dir, f'{self.iter_step}_valid.png'))
                        Image.fromarray(
                            np.clip(rgb.detach().cpu().numpy().reshape(H, W, 3) * 255., 0, 255).astype(np.uint8)).save(os.path.join(val_dir, f'{self.iter_step}_render.png'))

                loss_output = main_loss_dict

                invis_angle_loss = torch.tensor(0.0).cuda().float()

                obj_i = np.random.randint(1, num_objs)
                vis_info_obj_i = vis_info_list[obj_i]
                if len(vis_info_obj_i) > 0:
                    invis_angle_loss += self.get_invis_loss(vis_info_obj_i, mesh_obj_dict_list[obj_i], instance_gs_idxs[obj_i], save_dir=val_dir if self.iter_step % 200 == 0 else None, iter_step=self.iter_step)

                obj_i = 0
                vis_info_obj_i = vis_info_list[obj_i]
                if len(vis_info_obj_i) > 0:
                    invis_angle_loss += self.get_bg_loss(vis_info_obj_i, mesh_obj_dict_list[obj_i], instance_gs_idxs[obj_i], save_dir=val_dir if self.iter_step % 200 == 0 else None, iter_step=self.iter_step)

                loss += invis_angle_loss
                loss_output['invis_angle_loss'] = invis_angle_loss
                
                loss.backward()

                # calculate gradient norm
                total_norm = 0
                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                self.optimizer.step()
                if not torch.any(valid):
                    psnr = torch.tensor(0.0).float()
                else:
                    psnr = rend_util.get_psnr(rgb.reshape(-1, 3)[valid.reshape(-1)],
                                      rgb_gt.cuda().reshape(-1,3)[valid.reshape(-1)])
                # .module
                self.iter_step += 1

                if data_index % 20 == 0:

                    print("{0}_{1} [{2}/{3}] ({4}/{5}):".format(self.expname, self.timestamp, epoch, self.nepochs, data_index, self.n_batches), end=' ')
                    print("loss_main: {0:.4f}; psnr: {1:.4f}".format(loss_output['main_loss'].item(), psnr.item()))

                self.scheduler.step()

                if self.iter_step % 1000 == 0 and self.iter_step > 0:
                    print("saving ckpt...")
                    for obj_i in range(self.n_sem):
                        export_path = os.path.join(self.plots_dir, f"gauss_obj_{obj_i}.pt")
                        self.model.export_gs_pt(export_path, self.instance_gs_idxs[obj_i])

                    if self.train_dataset.test_split:
                        print("evaluating the model... ")
                        self.eval_test()

                if self.iter_step >= total_steps:
                    break
            if self.iter_step >= total_steps:
                break

        print("evaluating the model... ")
        self.eval_train()
        if self.train_dataset.test_split:
            self.eval_test()

        for obj_i in range(self.n_sem):
            export_path = os.path.join(self.plots_dir, f"gauss_obj_{obj_i}.ply")
            self.model.export_gs(export_path, self.instance_gs_idxs[obj_i])

        for obj_i in range(self.n_sem):
            export_path = os.path.join(self.plots_dir, f"gauss_obj_{obj_i}.pt")
            self.model.export_gs_pt(export_path, self.instance_gs_idxs[obj_i])

        self.eval_gs_load_train()
        if self.train_dataset.test_split:
            self.eval_gs_load_test()

        if self.use_wandb:
            wandb.finish()
        print('training over')

    def eval_test(self):
        from utils.eval_rgb import setup_eval_images
        compute_psnr, compute_ssim, compute_lpips = setup_eval_images()
        H, W = self.img_res
        num_test = len(self.train_dataset.test_mvps)
        self.model.eval()

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0

        with torch.no_grad():
            for test_i in tqdm(range(num_test)):
                pose = self.train_dataset.test_pose_all[test_i]
                K = self.train_dataset.test_intrinsics_all[test_i][:3, :3]
                rgb_gt = self.train_dataset.test_rgb_images[test_i]

                model_outputs = self.model.get_outputs(
                    pose=pose,
                    K=K,
                    H=H,
                    W=W,
                    camera_model="pinhole"
                )

                rgb_pred = model_outputs['rgb'].reshape(H, W, 3)
                rgb_gt = rgb_gt.reshape(H, W, 3).cuda()

                psnr = compute_psnr(rgb_gt, rgb_pred)
                ssim = compute_ssim(rgb_gt, rgb_pred)
                lpips = compute_lpips(rgb_gt, rgb_pred)

                avg_psnr += (psnr / num_test)
                avg_ssim += (ssim / num_test)
                avg_lpips += (lpips / num_test)

        print(f"Eval test result: PSNR: {avg_psnr:.4f} SSIM: {avg_ssim:.4f} LPIPS: {avg_lpips:.4f}")
        self.model.train()

    def eval_gs_load_test(self):

        gs_model = self.load_scene_gs_pt(self.plots_dir)

        gs_model.eval().cuda()
        gs_model.device = torch.device("cuda")
        num_test = len(self.train_dataset.test_mvps)

        from utils.eval_rgb import setup_eval_images
        compute_psnr, compute_ssim, compute_lpips = setup_eval_images()
        H, W = self.img_res

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0

        with torch.no_grad():
            for test_i in tqdm(range(num_test)):
                pose = self.train_dataset.test_pose_all[test_i]
                K = self.train_dataset.test_intrinsics_all[test_i][:3, :3]
                rgb_gt = self.train_dataset.test_rgb_images[test_i]

                model_outputs = gs_model.get_outputs(
                    pose=pose,
                    K=K,
                    H=H,
                    W=W,
                    camera_model="pinhole"
                )

                rgb_pred = model_outputs['rgb'].reshape(H, W, 3)
                rgb_gt = rgb_gt.reshape(H, W, 3).cuda()

                psnr = compute_psnr(rgb_gt, rgb_pred)
                ssim = compute_ssim(rgb_gt, rgb_pred)
                lpips = compute_lpips(rgb_gt, rgb_pred)

                avg_psnr += (psnr / num_test)
                avg_ssim += (ssim / num_test)
                avg_lpips += (lpips / num_test)

        print(f"Eval result from GS loading pt, testing set: PSNR: {avg_psnr:.4f} SSIM: {avg_ssim:.4f} LPIPS: {avg_lpips:.4f}")

    def eval_train(self):
        from utils.eval_rgb import setup_eval_images
        compute_psnr, compute_ssim, compute_lpips = setup_eval_images()
        H, W = self.img_res
        num_test = len(self.train_dataset.mvps)
        self.model.eval()

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0

        with torch.no_grad():
            for test_i in tqdm(range(num_test)):
                pose = self.train_dataset.pose_all[test_i]
                K = self.train_dataset.intrinsics_all[test_i][:3, :3]
                rgb_gt = self.train_dataset.rgb_images[test_i]

                model_outputs = self.model.get_outputs(
                    pose=pose,
                    K=K,
                    H=H,
                    W=W,
                    camera_model="pinhole"
                )

                rgb_pred = model_outputs['rgb'].reshape(H, W, 3)
                rgb_gt = rgb_gt.reshape(H, W, 3).cuda()

                psnr = compute_psnr(rgb_gt, rgb_pred)
                ssim = compute_ssim(rgb_gt, rgb_pred)
                lpips = compute_lpips(rgb_gt, rgb_pred)

                avg_psnr += (psnr / num_test)
                avg_ssim += (ssim / num_test)
                avg_lpips += (lpips / num_test)

        print(f"Eval train result: PSNR: {avg_psnr:.4f} SSIM: {avg_ssim:.4f} LPIPS: {avg_lpips:.4f}")
        self.model.train()

    def eval_gs_load_train(self):

        gs_model = self.load_scene_gs_pt(self.plots_dir)

        gs_model.eval().cuda()
        gs_model.device = torch.device("cuda")
        num_test = len(self.train_dataset.mvps)

        from utils.eval_rgb import setup_eval_images
        compute_psnr, compute_ssim, compute_lpips = setup_eval_images()
        H, W = self.img_res

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0

        with torch.no_grad():
            for test_i in tqdm(range(num_test)):
                pose = self.train_dataset.pose_all[test_i]
                K = self.train_dataset.intrinsics_all[test_i][:3, :3]
                rgb_gt = self.train_dataset.rgb_images[test_i]

                model_outputs = gs_model.get_outputs(
                    pose=pose,
                    K=K,
                    H=H,
                    W=W,
                    camera_model="pinhole"
                )

                rgb_pred = model_outputs['rgb'].reshape(H, W, 3)
                rgb_gt = rgb_gt.reshape(H, W, 3).cuda()

                psnr = compute_psnr(rgb_gt, rgb_pred)
                ssim = compute_ssim(rgb_gt, rgb_pred)
                lpips = compute_lpips(rgb_gt, rgb_pred)

                avg_psnr += (psnr / num_test)
                avg_ssim += (ssim / num_test)
                avg_lpips += (lpips / num_test)

        print(f"Eval result from GS loading pt, training set: PSNR: {avg_psnr:.4f} SSIM: {avg_ssim:.4f} LPIPS: {avg_lpips:.4f}")

    def load_scene_gs(self, load_dir):
        means_list = []
        opacities_list = []
        features_dc_list = []
        features_rest_list = []
        scales_list = []
        quats_list = []


        for obj_i in range(self.n_sem):
            gs_file_path = os.path.join(load_dir, f"gauss_obj_{obj_i}.ply")

            gs_dict = load_gaussian_from_ply(gs_file_path, device="cuda", dtype=torch.float32)


            sh_degree = gs_dict["sh_degree"]

            means_list.append(gs_dict["means"])
            opacities_list.append(gs_dict['opacities'])
            features_dc_list.append(gs_dict['features_dc'])
            features_rest_list.append(gs_dict['features_rest'])
            scales_list.append(gs_dict['scales'])
            quats_list.append(gs_dict['quats'])

        seed_gs = {
            'means': torch.cat(means_list, dim=0),
            'opacities': torch.cat(opacities_list, dim=0),
            'features_dc': torch.cat(features_dc_list, dim=0),
            'features_rest': torch.cat(features_rest_list, dim=0),
            'scales': torch.cat(scales_list, dim=0),
            'quats': torch.cat(quats_list, dim=0),
            'sh_degree': sh_degree
        }

        return GS(seed_gs=seed_gs)

    def load_scene_gs_pt(self, load_dir):
        means_list = []
        opacities_list = []
        features_dc_list = []
        features_rest_list = []
        scales_list = []
        quats_list = []


        for obj_i in range(self.n_sem):
            gs_file_path = os.path.join(load_dir, f"gauss_obj_{obj_i}.pt")

            gs_dict = torch.load(gs_file_path, map_location="cpu")


            sh_degree = gs_dict["sh_degree"]

            means_list.append(gs_dict["means"])
            opacities_list.append(gs_dict['opacities'])
            features_dc_list.append(gs_dict['shs_0'])
            features_rest_list.append(gs_dict['shs_rest'])
            scales_list.append(gs_dict['scales'])
            quats_list.append(gs_dict['quats'])

        seed_gs = {
            'means': torch.cat(means_list, dim=0),
            'opacities': torch.cat(opacities_list, dim=0),
            'features_dc': torch.cat(features_dc_list, dim=0),
            'features_rest': torch.cat(features_rest_list, dim=0),
            'scales': torch.cat(scales_list, dim=0),
            'quats': torch.cat(quats_list, dim=0),
            'sh_degree': sh_degree
        }

        return GS(seed_gs=seed_gs)

    def get_invis_loss(self, vis_info_list, obj_mesh_i_dict, instance_gs_idxs, save_dir=None, iter_step=None):


        loss_rgb = torch.tensor(0.0).cuda().float()

        gen_data_dict_list = vis_info_list
        # for data_idx in range(len(gen_data_dict_list)):
        data_idx = np.random.randint(len(gen_data_dict_list))
        gen_data_dict = gen_data_dict_list[data_idx]
        #
        # if gen_data_dict["source"] == "sdf":
        #     continue

        rgb_gt = gen_data_dict["rgb"]
        mask = gen_data_dict["mask"]
        pose = gen_data_dict["pose"]
        scale = gen_data_dict["scale"]
        fg_mask = gen_data_dict.get("fg_mask", None)
        sm_mask = gen_data_dict.get("sm_mask", None)

        H, W = rgb_gt.shape[:2]

        rgb_gt = torch.from_numpy(rgb_gt).float().reshape(H, W, 3).cuda()

        mask_original = torch.from_numpy(mask.copy()).reshape(H, W, 1).cuda()
        # mask = torch.from_numpy(mask).float().reshape(-1).cuda()
        if fg_mask is not None:
            # fg_mask = torch.from_numpy(fg_mask).float().reshape(-1).cuda()
            mask = fg_mask
        if gen_data_dict["source"] == "lama" and sm_mask is not None:
            # sm_mask = torch.from_numpy(sm_mask).float().reshape(-1).cuda()
            mask = sm_mask

        if gen_data_dict["source"] == "wonder3d" or gen_data_dict["source"] == "sdf":
            mask = binary_erosion(mask, iterations=np.random.randint(6, 10))
        elif gen_data_dict["source"] == "lama":
            mask = binary_dilation(mask, iterations=np.random.randint(1, 3))

        vis_mask = torch.from_numpy(mask).float().reshape(H, W, 1).cuda()

        diff_mask = torch.logical_and(mask_original, torch.logical_not(vis_mask)).reshape(H, W)

        x_offset = (np.random.random() - 0.5) / W
        y_offset = (np.random.random() - 0.5) / H

        pose = pose.cuda()

        cam_proj = get_camera_orthogonal_projection_matrix_offset(near=0.001, far=100.0, width=W, height=H, scale=scale, offsets=[x_offset, y_offset])
        cam_proj = torch.from_numpy(cam_proj).reshape(4, 4).float().cuda()

        mvp = cam_proj @ torch.inverse(pose)

        H_sr = H
        W_sr = W
        while W_sr % 8 != 0 or H_sr % 8 != 0:
            W_sr *= 2
            H_sr *= 2

        valid, _, mesh_depth = rasterize_mesh(obj_mesh_i_dict, mvp, self.glctx, (H_sr, W_sr), pose)

        if H_sr != H or W_sr != W:
            valid = torchvision.transforms.functional.resize(valid.unsqueeze(0), (H, W),
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(
                0)

            mesh_depth = mesh_depth.unsqueeze(-1)

            mesh_depth = mesh_depth.permute(2, 0, 1)
            mesh_depth = torchvision.transforms.functional.resize(mesh_depth, (H, W),
                                                                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            mesh_depth = mesh_depth.permute(1, 2, 0).squeeze(-1)

        K = torch.from_numpy(
            np.array([
                W / (2*scale), 0, W / 2 + x_offset * W,
                0, H / (2*scale), H / 2 + y_offset * H,
                0, 0, 1
            ], dtype=np.float32).reshape(3, 3)
        ).cuda().float()

        self.model.visible_gs_indices = instance_gs_idxs
        model_outputs = self.model.get_outputs(
            pose=pose,
            K=K,
            H=H,
            W=W,
            camera_model="ortho"
        )
        self.model.visible_gs_indices = None

        rgb = model_outputs['rgb'].reshape(H, W, 3)
        depth = model_outputs['depth'].reshape(H, W, 1)
        accumulation = model_outputs['accumulation'].reshape(H, W, 1)

        if gen_data_dict["source"] == "lama":
            rgb_gt_scale = torch.mean(rgb[diff_mask] / (rgb_gt[diff_mask] + 1e-6))
            rgb_gt_scale = torch.clip(rgb_gt_scale, 0, 2.0)
            rgb_gt[vis_mask.reshape(H, W) > 0] = rgb_gt[vis_mask.reshape(H, W) > 0] * rgb_gt_scale
            rgb_gt = torch.clip(rgb_gt, 0, 1.0)

        batch_gs = {
            "acm": valid.reshape(H, W, 1).float(),
            "mask": vis_mask.reshape(H, W, 1).float(),
            "mesh_depth": mesh_depth.reshape(H, W, 1).float(),
            "image": torch.cat([rgb_gt.reshape(H, W, 3), valid.reshape(H, W, 1).float()], dim=-1),
            # "image": rgb_gt.reshape(H, W, 3),
        }

        model_outputs = {
            'rgb': rgb,
            'depth': depth,
            'accumulation': accumulation,
            'background': model_outputs['background'],
        }

        main_loss_dict = self.model.get_loss_dict(model_outputs, batch_gs)

        loss = main_loss_dict["main_loss"] + main_loss_dict["scale_reg"]


        if save_dir is not None:
            with torch.no_grad():

                Image.fromarray(
                    np.clip(np.concatenate([rgb.detach().cpu().numpy(), valid.reshape(H, W, 1).cpu().float().numpy()], axis=-1) * 255., 0, 255).astype(np.uint8)).save(os.path.join(save_dir, f'{iter_step}_{data_idx}_render.png'))
                Image.fromarray(
                    np.clip(np.concatenate([rgb_gt.detach().cpu().numpy().reshape(H, W, 3), valid.reshape(H, W, 1).cpu().float().numpy()], axis=-1) * 255., 0, 255).astype(np.uint8)).save(os.path.join(save_dir, f'{iter_step}_{data_idx}_gt.png'))

        if gen_data_dict["source"] == "lama":
            loss *= 10.0

        return loss * 5.0

    def get_bg_loss(self, bg_info, bg_mesh_dict, instance_gs_idxs, save_dir=None, iter_step=None):

        loss_rgb = torch.tensor(0.0).cuda().float()

        # for bg_view_i in range(len(bg_info)):
        bg_view_i = np.random.randint(len(bg_info))
        bg_views_dict = bg_info[bg_view_i]


        rgb_gt = bg_views_dict["rgb"]
        normal = bg_views_dict["normal"]
        mask = bg_views_dict["mask"]
        pose = bg_views_dict["pose"]
        intrinsics = bg_views_dict["intrinsics"]

        if np.random.rand() < 0.5:
            mask = binary_dilation(mask, iterations=np.random.randint(1, 6))
        else:
            mask = binary_erosion(mask, iterations=np.random.randint(1, 6))

        H, W = normal.shape[:2]
        fx, fy, cx, cy = intrinsics

        cx = cx + np.random.rand() - 0.5
        cy = cy + np.random.rand() - 0.5

        rgb_gt = torch.from_numpy(rgb_gt).float().reshape(H, W, 3).cuda()
        vis_mask = torch.from_numpy(mask).float().reshape(H, W, 1).cuda()

        pose = pose.cuda()
        near = 0.001
        far = 100.0

        camera_projmat = get_camera_perspective_projection_matrix(fx, fy,
                                                                  cx,
                                                                  cy,
                                                                  H, W, near, far)
        camera_projmat = torch.from_numpy(camera_projmat).reshape(4, 4).float().cuda()
        mvp = camera_projmat @ torch.inverse(pose)


        H_sr = H
        W_sr = W
        while W_sr % 8 != 0 or H_sr % 8 != 0:
            W_sr *= 2
            H_sr *= 2

        valid, _, mesh_depth = rasterize_mesh(bg_mesh_dict, mvp, self.glctx, (H_sr, W_sr), pose)

        if H_sr != H or W_sr != W:
            valid = torchvision.transforms.functional.resize(valid.unsqueeze(0), (H, W),
                                                             interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(
                0)

            mesh_depth = mesh_depth.unsqueeze(-1)

            mesh_depth = mesh_depth.permute(2, 0, 1)
            mesh_depth = torchvision.transforms.functional.resize(mesh_depth, (H, W),
                                                                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            mesh_depth = mesh_depth.permute(1, 2, 0).squeeze(-1)

        K = torch.from_numpy(
            np.array([
                fx, 0, cx,
                0, fy, cy,
                0, 0, 1
            ], dtype=np.float32).reshape(3, 3)
        ).cuda().float()

        self.model.visible_gs_indices = instance_gs_idxs
        model_outputs = self.model.get_outputs(
            pose=pose,
            K=K,
            H=H,
            W=W,
            camera_model="pinhole"
        )
        self.model.visible_gs_indices = None

        rgb = model_outputs['rgb'].reshape(H, W, 3)
        depth = model_outputs['depth'].reshape(H, W, 1)
        accumulation = model_outputs['accumulation'].reshape(H, W, 1)

        batch_gs = {
            "acm": valid.reshape(H, W, 1).float(),
            "mask": vis_mask.reshape(H, W, 1).float(),
            "mesh_depth": mesh_depth.reshape(H, W, 1).float(),
            "image": torch.cat([rgb_gt.reshape(H, W, 3), valid.reshape(H, W, 1).float()], dim=-1),
            # "image": rgb_gt.reshape(H, W, 3),
        }

        model_outputs = {
            'rgb': rgb,
            'depth': depth,
            'accumulation': accumulation,
            'background': model_outputs['background'],
        }

        main_loss_dict = self.model.get_loss_dict(model_outputs, batch_gs)

        loss = main_loss_dict["main_loss"] + main_loss_dict["scale_reg"]

        # mse
        loss_rgb = loss

        if save_dir is not None:
            with torch.no_grad():

                Image.fromarray(
                    np.clip(np.concatenate([rgb.detach().cpu().numpy(), valid.reshape(H, W, 1).cpu().float().numpy()], axis=-1) * 255., 0, 255).astype(np.uint8)).save(os.path.join(save_dir, f'{iter_step}_{bg_view_i}_render.png'))
                Image.fromarray(
                    np.clip(np.concatenate([rgb_gt.detach().cpu().numpy().reshape(H, W, 3), valid.reshape(H, W, 1).cpu().float().numpy()], axis=-1) * 255., 0, 255).astype(np.uint8)).save(os.path.join(save_dir, f'{iter_step}_{bg_view_i}_gt.png'))


        return loss_rgb * 1.0

    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, seg_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift

        seg_map = model_outputs['semantic_values'].reshape(batch_size, num_samples)
        seg_gt = seg_gt.to(seg_map.device)

        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'seg_gt': seg_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'seg_map': seg_map,
        }

        return plot_data

    def mask_filter(self, verts, faces, obj_i):
        mesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(mesh_obj.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(mesh_obj.faces).int().to("cuda").contiguous()
        }

        mvps = self.train_dataset.mvps
        resolution = self.img_res
        triangle_vis = torch.zeros(mesh_obj.faces.shape[0], dtype=torch.bool)

        for camera_i in range(len(mvps)):
            mvp = mvps[camera_i].to("cuda")
            valid, triangle_id, depth = rasterize_mesh(mesh_dict, mvp, self.glctx, resolution)

            instance_mask_i = self.train_dataset.semantic_images[camera_i].int().to("cuda")
            valid_instance = instance_mask_i == obj_i

            valid = torch.logical_and(valid, valid_instance.reshape(valid.shape))

            triangle_vis_frame_i = torch.unique(triangle_id[valid]).reshape(-1)
            triangle_vis[triangle_vis_frame_i] = True

        triangle_vis = triangle_vis.cpu().numpy()
        vertices_seen_indices = np.unique(faces[triangle_vis].reshape(-1))

        edges = mesh_obj.edges_sorted.reshape((-1, 2))
        components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')

        components = [comp for comp in components if len(np.intersect1d(comp, vertices_seen_indices)) > 0]
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

        return verts_lcc, faces_lcc, verts_map, faces_map

