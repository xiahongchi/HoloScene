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
    load_tex_dict_from_path, rasterize_mesh_dict_list_with_uv
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
from pytorch3d.io import save_obj
from sklearn.neighbors import NearestNeighbors
import torchvision
from model.network import ColorImplicitNetworkSingle

class HoloSceneTrainTextureRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.description = kwargs['description']
        self.use_wandb = kwargs['use_wandb']
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

        self.max_total_iters = 5000
        self.ds_len = len(self.train_dataset)
        self.n_sem = len(self.train_dataset.label_mapping)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        # if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
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

                translation_dict_path = os.path.abspath(os.path.join(old_plots_folder, 'coarse_recon_obj_sim_mesh/translation_dict.pkl'))
                new_ln_path = os.path.abspath(os.path.join(self.plots_dir, 'translation_dict.pkl'))
                if os.path.exists(translation_dict_path):
                    os.symlink(translation_dict_path, new_ln_path)
                
                graph_node_dict_path = os.path.abspath(os.path.join(old_plots_folder, 'coarse_recon_obj_sim_mesh/graph_node_dict.pkl'))
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

            # # visiualize gradient
            # wandb.watch(self.model, self.optimizer)

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

        for obj_i in range(num_objs):

            self.model = ColorImplicitNetworkSingle()

            if torch.cuda.is_available():
                self.model.cuda()

            # The MLP and hash grid should have different learning rates
            self.lr = self.conf.get_float('train.learning_rate')
            self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)

            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.grid_parameters()),
                 'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.mlp_parameters()),
                 'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)

            # Exponential learning rate scheduler
            decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
            total_iterations = self.max_total_iters if obj_i == 0 else self.max_total_iters // 10
            decay_steps = total_iterations
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1. / decay_steps))

            self.iter_step = 0
            val_dir = os.path.join(self.plots_dir, f'val_{obj_i}')
            os.makedirs(val_dir, exist_ok=True)

            mesh_obj_i = all_meshes[obj_i]

            mesh_dict = {
                'vertices': F.pad(
                    torch.from_numpy(mesh_obj_i.vertices).float().to("cuda").contiguous(),
                    pad=(0, 1), value=1.0, mode='constant'),
                'pos_idx': torch.from_numpy(mesh_obj_i.faces).int().to("cuda").contiguous(),
                'vertices_world': torch.from_numpy(mesh_obj_i.vertices).float().to("cuda").contiguous(),
            }

            vis_info_obj_i = vis_info_list[obj_i]

            self.train_dataset.sampling_class_id = obj_i

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

                    cx = cx + np.random.rand() - 0.5
                    cy = cy + np.random.rand() - 0.5

                    rgb_gt = ground_truth["rgb"].cuda().reshape(H, W, 3)
                    instance_mask = ground_truth["mask"].cuda().reshape(-1).int() == obj_i

                    camera_projmat = get_camera_perspective_projection_matrix(fx, fy, cx, cy, H, W, near, far)
                    camera_projmat = torch.from_numpy(camera_projmat).reshape(4, 4).float().cuda()

                    mvp = camera_projmat @ torch.inverse(pose)

                    H_sr = H
                    W_sr = W
                    while W_sr % 8 != 0 or H_sr % 8 != 0:
                        W_sr *= 2
                        H_sr *= 2

                    valid, triangle_id, vertices_world, barys = \
                        rasterize_mesh_return_pixel_vert_and_bary(mesh_dict, mvp, self.glctx, (H_sr, W_sr))
                    # print(f"valid {valid.shape}, triangle_id {triangle_id.shape}, vertices_world {vertices_world.shape}, barys {barys.shape}")

                    if H_sr != H or W_sr != W:
                        valid = torchvision.transforms.functional.resize(valid.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
                        triangle_id = torchvision.transforms.functional.resize(triangle_id.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
                        vertices_world = vertices_world.permute(2, 3, 0, 1)
                        vertices_world = torchvision.transforms.functional.resize(vertices_world, (H, W), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                        vertices_world = vertices_world.permute(2, 3, 0, 1)
                        barys = barys.permute(2, 0, 1)
                        barys = torchvision.transforms.functional.resize(barys, (H, W), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                        barys = barys.permute(1, 2, 0)

                        # print(f"resized valid {valid.shape}, triangle_id {triangle_id.shape}, vertices_world {vertices_world.shape}, barys {barys.shape}")


                    # debug
                    if self.iter_step % 200 == 0:
                        with torch.no_grad():
                            Image.fromarray(
                                np.clip(rgb_gt.detach().cpu().numpy().reshape(H, W, 3) * 255., 0, 255).astype(np.uint8)).save(os.path.join(val_dir, f'{self.iter_step}_gt.png'))
                            Image.fromarray(
                                valid.detach().cpu().numpy().reshape(H, W).astype(np.uint8) * 255).save(os.path.join(val_dir, f'{self.iter_step}_valid.png'))
                            Image.fromarray(
                                instance_mask.detach().cpu().numpy().reshape(H, W).astype(np.uint8) * 255).save(
                                os.path.join(val_dir, f'{self.iter_step}_instance_mask.png'))

                    # assert False
                    rgb_values = torch.ones_like(rgb_gt).cuda()
                    if not torch.any(valid):
                        loss_rgb = torch.tensor(0.0).cuda().float()
                        continue
                    else:
                        valid = valid.reshape(-1)
                        valid = torch.logical_and(valid, instance_mask)
                        if not torch.any(valid):
                            loss_rgb = torch.tensor(0.0).cuda().float()
                            continue
                        triangle_id = triangle_id.reshape(-1)
                        vertices_world = vertices_world.reshape(-1, 3, 3)
                        barys = barys.reshape(-1, 3, 1)
                        vertices_pix = vertices_world[:, 0, :] * barys[:, 0, :] + vertices_world[:, 1, :] * barys[:, 1, :] + vertices_world[:, 2, :] * barys[:, 2, :]
                        vertices_pix = vertices_pix.reshape(-1, 3)
                        vertices_pix = vertices_pix[valid]

                        rgb_pix_masked = self.model(vertices_pix)
                        rgb_gt_masked = rgb_gt.reshape(-1, 3)[valid]
                        #     mse
                        loss_rgb = torch.mean((rgb_pix_masked - rgb_gt_masked) ** 2)
                        rgb_values[valid.reshape(H, W)] = rgb_pix_masked

                    if self.iter_step % 200 == 0:
                        with torch.no_grad():
                            Image.fromarray(
                                np.clip(rgb_values.detach().cpu().numpy().reshape(H, W, 3) * 255., 0, 255).astype(np.uint8)).save(os.path.join(val_dir, f'{self.iter_step}_render.png'))

                    loss = torch.tensor(0.0).cuda().float()
                    loss += loss_rgb

                    loss_output = {}
                    loss_output['rgb_loss'] = loss_rgb

                    invis_angle_loss = torch.tensor(0.0).cuda().float()

                    if obj_i > 0:
                        if len(vis_info_obj_i) > 0:
                            invis_angle_loss += self.get_invis_loss(vis_info_obj_i, mesh_dict, save_dir=val_dir if self.iter_step % 200 == 0 else None, iter_step=self.iter_step)

                    else:
                        if len(vis_info_obj_i) > 0:

                            invis_angle_loss += self.get_bg_loss(vis_info_obj_i, mesh_dict)

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
                        psnr = rend_util.get_psnr(rgb_values.reshape(-1, 3)[valid],
                                          ground_truth['rgb'].cuda().reshape(-1,3)[valid])
                    # .module
                    self.iter_step += 1

                    if data_index % 20 == 0:

                        print("{0}_{1} [{2}/{3}] ({4}/{5}):".format(self.expname, self.timestamp, epoch, self.nepochs, data_index, self.n_batches), end=' ')
                        print("loss_rgb: {0:.4f}; psnr: {1:.4f}; loss_invis: {2:.4f}".format(loss_rgb.item(), psnr.item(), invis_angle_loss.item()))

                        if self.use_wandb:
                            for k, v in loss_output.items():
                                wandb.log({f'Loss/{k}': v.item()}, self.iter_step)

                            wandb.log({'Statistics/psnr': psnr.item()}, self.iter_step)
                            wandb.log({'Statistics/total_norm': total_norm}, self.iter_step)
                            wandb.log({'Statistics/loss_invis': invis_angle_loss.item()}, self.iter_step)

                    self.scheduler.step()

                    if self.iter_step >= total_iterations:
                        break
                if self.iter_step >= total_iterations:
                    break

            self.export_mesh_texture(mesh_obj_i, obj_i)

        if self.use_wandb:
            wandb.finish()
        print('training over')
        print("saving dir: ", self.plots_dir)

    def get_invis_loss(self, vis_info_list, obj_mesh_i_dict, save_dir=None, iter_step=None):


        loss_rgb = torch.tensor(0.0).cuda().float()

        gen_data_dict_list = vis_info_list
        for data_idx in range(len(gen_data_dict_list)):
            gen_data_dict = gen_data_dict_list[data_idx]
            #
            # if gen_data_dict["source"] == "sdf":
            #     continue

            rgb = gen_data_dict["rgb"]
            mask = gen_data_dict["mask"]
            pose = gen_data_dict["pose"]
            scale = gen_data_dict["scale"]
            subset_idxs = gen_data_dict["obj_idxs"]
            fg_mask = gen_data_dict.get("fg_mask", None)
            sm_mask = gen_data_dict.get("sm_mask", None)

            H, W = rgb.shape[:2]

            rgb = torch.from_numpy(rgb).float().reshape(-1, 3).cuda()
            mask = torch.from_numpy(mask).float().reshape(-1).cuda()
            if fg_mask is not None:
                fg_mask = torch.from_numpy(fg_mask).float().reshape(-1).cuda()
                mask = fg_mask
            if gen_data_dict["source"] == "lama" and sm_mask is not None:
                sm_mask = torch.from_numpy(sm_mask).float().reshape(-1).cuda()
                mask = sm_mask

            if gen_data_dict["source"] == "wonder3d" or gen_data_dict["source"] == "sdf":
                mask = torch.from_numpy(binary_erosion(mask.cpu().numpy(), iterations=np.random.randint(6, 10))).float().cuda()
            elif gen_data_dict["source"] == "lama":
                mask = torch.from_numpy(binary_dilation(mask.cpu().numpy(), iterations=np.random.randint(1, 5))).float().cuda()


            pose = pose.cuda()

            cam_proj = get_camera_orthogonal_projection_matrix_offset(near=0.001, far=100.0, width=W, height=H, scale=scale)
            cam_proj = torch.from_numpy(cam_proj).reshape(4, 4).float().cuda()

            mvp = cam_proj @ torch.inverse(pose)

            obj_i = subset_idxs[0]

            valid, triangle_id, vertices_world, barys = rasterize_mesh_return_pixel_vert_and_bary(obj_mesh_i_dict,
                                                                                                  mvp, self.glctx,
                                                                                                  (H, W))

            valid = valid.reshape(-1)
            valid = torch.logical_and(valid, mask)

            if not torch.any(valid):
                continue

            triangle_id = triangle_id.reshape(-1)
            vertices_world = vertices_world.reshape(-1, 3, 3)
            barys = barys.reshape(-1, 3, 1)
            vertices_pix = vertices_world[:, 0, :] * barys[:, 0, :] + vertices_world[:, 1, :] * barys[:, 1,
                                                                                                :] + vertices_world[:, 2,
                                                                                                     :] * barys[:, 2, :]
            vertices_pix = vertices_pix.reshape(-1, 3)
            vertices_pix = vertices_pix[valid]

            rgb_pix_masked = self.model(vertices_pix)
            rgb_gt_masked = rgb.reshape(-1, 3)[valid]

            # mse
            loss_rgb += torch.mean(((rgb_pix_masked - (rgb_gt_masked + (torch.rand(rgb_gt_masked.shape[0], 3).cuda().float() - 0.5) * 0.0 ) ) ** 2) * torch.rand(rgb_gt_masked.shape[0]).cuda().float().reshape(-1, 1) + 0.5)

            if save_dir is not None:
                with torch.no_grad():
                    rgb_pix = torch.ones((H, W, 3))
                    rgb_pix[valid.reshape(H, W).cpu()] = rgb_pix_masked.cpu()

                    Image.fromarray(
                        np.clip(np.concatenate([rgb_pix.detach().cpu().numpy(), valid.reshape(H, W, 1).cpu().float().numpy()], axis=-1) * 255., 0, 255).astype(np.uint8)).save(os.path.join(save_dir, f'{iter_step}_{data_idx}_render.png'))
                    Image.fromarray(
                        np.clip(np.concatenate([rgb.detach().cpu().numpy().reshape(H, W, 3), valid.reshape(H, W, 1).cpu().float().numpy()], axis=-1) * 255., 0, 255).astype(np.uint8)).save(os.path.join(save_dir, f'{iter_step}_{data_idx}_gt.png'))




        return (loss_rgb / len(gen_data_dict_list)) * 5.0

    def get_bg_loss(self, bg_info, bg_mesh_dict):

        loss_rgb = torch.tensor(0.0).cuda().float()

        for bg_view_i in range(len(bg_info)):
            bg_views_dict = bg_info[bg_view_i]


            rgb = bg_views_dict["rgb"]
            normal = bg_views_dict["normal"]
            mask = bg_views_dict["mask"]
            pose = bg_views_dict["pose"]
            intrinsics = bg_views_dict["intrinsics"]

            mask = binary_erosion(mask, iterations=np.random.randint(1, 6))

            H, W = normal.shape[:2]
            fx, fy, cx, cy = intrinsics

            rgb = torch.from_numpy(rgb).float().reshape(-1, 3).cuda()
            mask = torch.from_numpy(mask).float().reshape(-1).cuda()

            pose = pose.cuda()
            near = 0.001
            far = 100.0

            camera_projmat = get_camera_perspective_projection_matrix(fx, fy,
                                                                      cx+np.random.rand() - 0.5,
                                                                      cy+np.random.rand() - 0.5,
                                                                      H, W, near, far)
            camera_projmat = torch.from_numpy(camera_projmat).reshape(4, 4).float().cuda()
            mvp = camera_projmat @ torch.inverse(pose)

            valid, triangle_id, vertices_world, barys = rasterize_mesh_return_pixel_vert_and_bary(bg_mesh_dict, mvp,
                                                                                                  self.glctx, (H, W))

            valid = valid.reshape(-1)
            valid = torch.logical_and(valid, mask)
            triangle_id = triangle_id.reshape(-1)
            vertices_world = vertices_world.reshape(-1, 3, 3)
            barys = barys.reshape(-1, 3, 1)
            vertices_pix = vertices_world[:, 0, :] * barys[:, 0, :] + vertices_world[:, 1, :] * barys[:, 1,
                                                                                                :] + vertices_world[:, 2,
                                                                                                     :] * barys[:, 2, :]
            vertices_pix = vertices_pix.reshape(-1, 3)
            vertices_pix = vertices_pix[valid]

            rgb_pix_masked = self.model(vertices_pix)
            rgb_gt_masked = rgb.reshape(-1, 3)[valid]
            # mse
            loss_rgb += torch.mean(((rgb_pix_masked - rgb_gt_masked) ** 2) * (torch.rand(rgb_gt_masked.shape[0]).cuda().float().reshape(-1, 1) + 0.5))

        return loss_rgb / len(bg_info)

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

    @torch.no_grad()
    def export_mesh_texture(self, obj_i_mesh, obj_i):
        num_objs = self.n_sem
        texture_res = 2048

            # obj_i_mesh = self.all_meshes[obj_i]
        verts = torch.from_numpy(obj_i_mesh.vertices).float()
        faces = torch.from_numpy(obj_i_mesh.faces).long()

        v_np = obj_i_mesh.vertices
        f_np = obj_i_mesh.faces
        v = torch.from_numpy(v_np).float().cuda()
        f = torch.from_numpy(f_np).int().cuda()


        print("Generate UV coordinates for mesh: ", obj_i)
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np * 10, f_np)
        atlas.generate()
        _, ft_np, vt_np = atlas[0]
        print("Finish generation of UV coordinates")

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()

        # padding
        uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (texture_res, texture_res))
        texture_valid = (rast[..., -1] > 0).reshape(texture_res, texture_res).cpu()

        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
        vt = vt.cpu()
        ft = ft.cpu()

        xyzs = xyzs.view(-1, 3).cpu()
        mask = texture_valid.reshape(-1)

        tex_color = torch.zeros(texture_res, texture_res, 3).reshape(-1, 3).float()

        if mask.any():
            xyzs_masked = xyzs[mask]
            color_all = []

            for _xyzs in tqdm(torch.split(xyzs_masked, 1024, dim=0), total=len(xyzs_masked) // 1024 + 1):
                _color = self.model(_xyzs.cuda().float())
                color_all.append(_color.detach().cpu())

            tex_color_masked = torch.cat(color_all, dim=0)
            tex_color[mask] = tex_color_masked

        tex_color = tex_color.reshape(texture_res, texture_res, 3)
        vt_export = vt.clone()
        vt_export[:, 1] = 1 - vt_export[:, 1]

        if mask.any():
            mask = mask.reshape(texture_res, texture_res).numpy()
            inpaint_region = binary_dilation(mask, iterations=8)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=4)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            tex_color[tuple(inpaint_coords.T)] = tex_color[tuple(search_coords[indices[:, 0]].T)]

        save_obj(
            os.path.join(self.plots_dir, f"surface_{obj_i}.obj"),
            v.cpu(),
            f.cpu(),
            verts_uvs=vt_export,
            faces_uvs=ft,
            texture_map=tex_color
        )

