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
import gc

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
    generate_traverse_seq, coarse_recon, make_wonder3D_cameras, marching_cubes_from_sdf, marching_cubes_from_sdf_center_scale,
    simplify_mesh, detect_collision, marching_cubes_from_sdf_center_scale_single_object, rasterize_mesh_with_uv,
    generate_color_from_model_and_mesh, grid_sample_3d, grid_sample_3d_with_channels, clean_mesh_floaters_adjust,
    marching_cubes_from_sdf_center_scale_rm_intersect, remesh, create_scene_graph_from_meshes, 
    convert_parent_child_to_adjacency_list
)
from datasets.ns_dataset import extract_graph_node_properties
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
from model.network import ObjectSDFNetwork
from model.loss import compute_scale_and_shift_batch
import copy
import open3d as o3d
from utils.sim import start_simulation_app, sim_validation, sim_scene

class HoloSceneTrainPostRunner():
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
        self.glctx = dr.RasterizeGLContext()

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

        # save checkpoints

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

        print("training...")

        print('[INFO]: shell command : {0}'.format(' '.join(sys.argv)))

        print('[INFO]: Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        self.invis_loss_conf = self.conf.get_config('invis_loss')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0} EPOCHS'.format(self.nepochs))

        start_simulation_app()

        if len(self.train_dataset.label_mapping) > 0:
            # a hack way to let network know how many categories, so don't need to manually set in config file
            self.conf['model']['implicit_network']['d_out'] = len(self.train_dataset.label_mapping)
            print('RUNNING FOR {0} CLASSES'.format(len(self.train_dataset.label_mapping)))

        self.n_sem = self.conf.get_int('model.implicit_network.d_out')
        assert self.n_sem == len(self.train_dataset.label_mapping)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8,
                                                            pin_memory=True)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(
            conf=conf_model,
            plots_dir=self.plots_dir,
            graph_node_dict=self.train_dataset.graph_node_dict,
            ft_folder=self.ft_folder,
            num_images=len(self.train_dataset.mvps)
        )

        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(
            **self.conf.get_config('loss'),
        )

        # The MLP and hash grid should have different learning rates
        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)

        self.optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()),
                'lr': self.lr * self.lr_factor_for_grid},
            {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                list(self.model.rendering_network.parameters()),
                'lr': self.lr},
            {'name': 'density', 'params': list(self.model.density.parameters()),
                'lr': self.lr},
        ], betas=(0.9, 0.99), eps=1e-15)

        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        single_gpu_mode = True

        self.start_epoch = 0
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')
            self.finetune_folder = os.path.join(self.expdir, timestamp)

            print('[INFO]: Loading pretrained model from {}'.format(old_checkpnts_dir))
            model_state_load_path = os.path.abspath(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            saved_model_state = torch.load(model_state_load_path)
            model_state_dict = {}
            if single_gpu_mode:
                for k, v in saved_model_state["model_state_dict"].items():
                    model_state_dict[k.replace('module.', '')] = v
            else:
                model_state_dict = saved_model_state["model_state_dict"]
            self.model.load_state_dict(model_state_dict)
            self.start_epoch = saved_model_state['epoch']
            self.ckpt_dict = {
                'model': copy.deepcopy(model_state_dict),
            }
            opt_state_load_path = os.path.abspath(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            data = torch.load(opt_state_load_path)
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.ckpt_dict['optimizer'] = copy.deepcopy(data["optimizer_state_dict"])

            sche_state_load_path = os.path.abspath(os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            data = torch.load(sche_state_load_path)
            self.scheduler.load_state_dict(data["scheduler_state_dict"])
            self.ckpt_dict['scheduler'] = copy.deepcopy(data["scheduler_state_dict"])

            os.makedirs(os.path.join(self.checkpoints_path, self.model_params_subdir), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints_path, self.optimizer_params_subdir), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoints_path, self.scheduler_params_subdir), exist_ok=True)


            for ckpt_name in sorted(os.listdir(os.path.dirname(model_state_load_path))):
                ckpt_path_old = os.path.join(os.path.dirname(model_state_load_path), ckpt_name)
                ckpt_path_ln = os.path.join(self.checkpoints_path, self.model_params_subdir, ckpt_name)
                if not os.path.exists(ckpt_path_ln):
                    os.symlink(ckpt_path_old, ckpt_path_ln)

            for ckpt_name in sorted(os.listdir(os.path.dirname(opt_state_load_path))):
                ckpt_path_old = os.path.join(os.path.dirname(opt_state_load_path), ckpt_name)
                ckpt_path_ln = os.path.join(self.checkpoints_path, self.optimizer_params_subdir, ckpt_name)
                if not os.path.exists(ckpt_path_ln):
                    os.symlink(ckpt_path_old, ckpt_path_ln)

            for ckpt_name in sorted(os.listdir(os.path.dirname(sche_state_load_path))):
                ckpt_path_old = os.path.join(os.path.dirname(sche_state_load_path), ckpt_name)
                ckpt_path_ln = os.path.join(self.checkpoints_path, self.scheduler_params_subdir, ckpt_name)
                if not os.path.exists(ckpt_path_ln):
                    os.symlink(ckpt_path_old, ckpt_path_ln)

            # continue training need copy mesh files from old folder
            old_plots_folder = os.path.join(self.finetune_folder, 'plots')
            mesh_str = f'surface_{self.start_epoch}_*'
            cmd = f'cp {old_plots_folder}/{mesh_str} {self.plots_dir}'
            os.system(cmd)
            cmd = f'cp -r {old_plots_folder}/bbox {self.plots_dir}'
            os.system(cmd)

            vis_info_path = os.path.join(old_plots_folder, 'vis_info.pkl')
            if os.path.exists(vis_info_path):
                # cmd = f'cp {vis_info_path} {self.plots_dir}'
                vis_info_link_path = os.path.join(self.plots_dir, 'vis_info.pkl')
                if not os.path.exists(vis_info_link_path):
                    os.symlink(os.path.abspath(vis_info_path), vis_info_link_path)

            for obj_i in range(self.n_sem):
                old_vis_info_i_path = os.path.abspath(os.path.join(old_plots_folder, f'vis_info_{obj_i}.pkl'))
                if os.path.exists(old_vis_info_i_path):
                    # cmd = f'cp {old_vis_info_i_path} {self.plots_dir}/vis_info_{obj_i}.pkl'
                    # os.system(cmd)
                    os.symlink(os.path.abspath(old_vis_info_i_path), os.path.join(self.plots_dir, f'vis_info_{obj_i}.pkl'))

            bg_info_path = os.path.join(old_plots_folder, 'bg_info.pkl')
            if os.path.exists(bg_info_path):
                # cmd = f'cp {bg_info_path} {self.plots_dir}'
                bg_info_link_path = os.path.join(self.plots_dir, 'bg_info.pkl')
                if not os.path.exists(bg_info_link_path):
                    os.symlink(os.path.abspath(bg_info_path), bg_info_link_path)


        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

        self.add_objectvio_iter = self.conf.get_int('train.add_objectvio_iter', default=100000)

        print('Loading mv diffusion pipeline ...')
        config_mvdiffusion = 'confs/mvdiffusion-joint.yaml'
        self.config_mv = OmegaConf.load(config_mvdiffusion)
        self.mv_pipeline = load_wonder3d_pipeline(self.config_mv).to("cuda")

        print("loading SR model...")
        rrdb_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=rrdb_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device="cuda"
        )


        print("loading lama model...")
        lama_config_path = "./lama/configs/prediction/default.yaml"
        lama_checkpoint_path = './lama/big-lama/'

        self.lama_model, self.lama_predict_config = load_model(lama_config_path, lama_checkpoint_path)


        print("loading omnidata normal model...")
        normal_model_path = "./omnidata_dpt_normal_v2.ckpt"
        self.omnidata_normal_model = load_normal_model(normal_model_path)

    def move_foundation_model_to_cpu(self):
        # print gpu memory usage with format as "allocated: 0.0 GB, reserved: 0.0 GB, total: 0.0 GB"
        print("allocated: {:.1f} GB, reserved: {:.1f} GB, total: {:.1f} GB".format(
            torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_reserved() / 1024 ** 3,
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

        self.mv_pipeline = self.mv_pipeline.to("cpu")
        self.upsampler.model = self.upsampler.model.to("cpu")
        self.lama_model = self.lama_model.to("cpu")
        self.omnidata_normal_model = self.omnidata_normal_model.to("cpu")

        torch.cuda.empty_cache()

        print("allocated: {:.1f} GB, reserved: {:.1f} GB, total: {:.1f} GB".format(
            torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_reserved() / 1024 ** 3,
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

    
    def move_foundation_model_to_gpu(self):
        # print gpu memory usage with format as "allocated: 0.0 GB, reserved: 0.0 GB, total: 0.0 GB"
        print("allocated: {:.1f} GB, reserved: {:.1f} GB, total: {:.1f} GB".format(
            torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_reserved() / 1024 ** 3,
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

        self.mv_pipeline = self.mv_pipeline.to("cuda")
        self.upsampler.model = self.upsampler.model.to("cuda")
        self.lama_model = self.lama_model.to("cuda")
        self.omnidata_normal_model = self.omnidata_normal_model.to("cuda")

        torch.cuda.empty_cache()

        print("allocated: {:.1f} GB, reserved: {:.1f} GB, total: {:.1f} GB".format(
            torch.cuda.memory_allocated() / 1024 ** 3, torch.cuda.memory_reserved() / 1024 ** 3,
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

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

        infos = json.loads(json.dumps(self.conf))
        wandb.init(
            config=infos,
            project=self.conf.get_string('wandb.project_name'),
            name='{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()),
        )

        self.iter_step = self.start_epoch * len(self.train_dataset)
        print(f'Start epoch: {self.start_epoch}, iter_step: {self.iter_step}')

        all_meshes = self.instance_meshes_post_pruning(self.start_epoch)
        for mesh_i, mesh in enumerate(all_meshes):
            trimesh.exchange.export.export_mesh(mesh, os.path.join(self.plots_dir, f'surface_{self.start_epoch}_{mesh_i}.ply'))

        self.original_bbox_dict = self.get_obj_bbox(self.start_epoch)
        all_meshes = self.instance_meshes_post_pruning_selected(self.start_epoch)
        for mesh_i, mesh in enumerate(all_meshes):
            trimesh.exchange.export.export_mesh(mesh, os.path.join(self.plots_dir, f'surface_{self.start_epoch}_{mesh_i}.ply'))
       
        if self.train_dataset.graph_node_dict is None:
            graph_node_dict = self.update_graph_node_dict(all_meshes)
            self.train_dataset.graph_node_dict = graph_node_dict
            self.model.graph_node_dict = graph_node_dict

        print(f"allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB, reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.1f} GB, total: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        self.mesh_coarse_points_collisions_dict = {}
        self.mesh_coarse_recon_dict = {}

        # first reconstruct the background
        if os.path.exists(os.path.join(self.finetune_folder, "plots", "coarse_recon_obj_0.ply")):
            # if we have that, load it
            load_base_dir = os.path.join(self.finetune_folder, "plots")
            self.mesh_coarse_recon_dict[0] = trimesh.exchange.load.load_mesh(
                load_base_dir+"/coarse_recon_obj_0.ply")
            self.mesh_coarse_points_collisions_dict[0] = pickle.load(
                open(load_base_dir + "/coarse_recon_obj_collision_pts_sdf_0.pkl", 'rb'))
            load_base_dir = os.path.abspath(load_base_dir)
            ply_link_path = os.path.join(self.plots_dir, 'coarse_recon_obj_0.ply')
            pkl_link_path = os.path.join(self.plots_dir, 'coarse_recon_obj_collision_pts_sdf_0.pkl')
            if not os.path.exists(ply_link_path):
                os.symlink(os.path.join(load_base_dir, 'coarse_recon_obj_0.ply'), ply_link_path)
            if not os.path.exists(pkl_link_path):
                os.symlink(os.path.join(load_base_dir, 'coarse_recon_obj_collision_pts_sdf_0.pkl'), pkl_link_path)
        else:
            bg_info_path = os.path.join(self.plots_dir, 'bg_info.pkl')
            if os.path.exists(bg_info_path):
                with open(bg_info_path, 'rb') as f:
                    self.bg_info = pickle.load(f)
            else:
                print("generating bg_info from lama...")
                self.bg_info = self.background_inpainting()
                with open(bg_info_path, 'wb') as f:
                    pickle.dump(self.bg_info, f)

            self.move_foundation_model_to_cpu()
            torch.cuda.empty_cache()
            self.background_reconstruction()
            self.move_foundation_model_to_gpu()

        vis_info_path = os.path.join(self.plots_dir, 'vis_info.pkl')
        self.vis_info = self.generative_sampling()

    def calculate_invisible_loss(self, view_dict_list, local_model, data_idx=None, near=0.001, far=1.0):


        gen_data_dict_list = view_dict_list
        if data_idx is None:
            data_idx = np.random.choice(len(gen_data_dict_list))
        gen_data_dict = gen_data_dict_list[data_idx]

        rgb = gen_data_dict["rgb"]
        normal = gen_data_dict["normal"]
        mask = gen_data_dict["mask"]
        nm_mask = gen_data_dict.get('nm_mask', None)
        depth = gen_data_dict.get('depth', None)
        depth_mask = gen_data_dict.get('depth_mask', None)
        pose = gen_data_dict["pose"]
        scale = gen_data_dict["scale"]
        subset_idxs = gen_data_dict["obj_idxs"]
        front = gen_data_dict["front"]
        bg_color = gen_data_dict["bg_color"]
        fg_mask = gen_data_dict.get("fg_mask", None)
        image_source = gen_data_dict["source"]
        sm_mask = gen_data_dict.get("sm_mask", None)
        diff_mask = gen_data_dict.get("diff_mask", None)

        H, W = normal.shape[:2]

        loss_lambda = gen_data_dict.get("loss_lambda", 1.0)

        rgb = torch.from_numpy(rgb).float().reshape(-1, 3)
        normal = torch.from_numpy(normal).float().reshape(-1, 3)
        mask = torch.from_numpy(mask).float().reshape(-1)
        if nm_mask is not None:
            nm_mask = torch.from_numpy(nm_mask).float().reshape(-1)
        if sm_mask is not None:
            sm_mask = torch.from_numpy(sm_mask).float().reshape(-1)

        pose = pose.cuda()

        num_rays = self.invis_loss_conf.get("num_rays", 256)


        if fg_mask is not None:
            ray_bundle_size = min(num_rays, np.count_nonzero(fg_mask.reshape(-1)))
        else:
            ray_bundle_size = num_rays

        ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near=0.001, pose=pose.cpu().clone(), scale=scale)
        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        # randomly select ray_bundle_size rays

        if diff_mask is not None:

            if fg_mask is not None:
                selected_indices = np.random.choice(np.nonzero(fg_mask.reshape(-1))[0], int(ray_bundle_size * 0.7), replace=False)
            else:
                selected_indices = np.random.choice(ray_origins.shape[0], int(ray_bundle_size * 0.7), replace=False)

            selected_indices_diff_mask = np.random.choice(
                np.nonzero(diff_mask.reshape(-1))[0],
                min(int(ray_bundle_size * 0.3), np.count_nonzero(diff_mask.reshape(-1))), replace=False)

            # print("selected_indices: ", selected_indices.shape[0])
            # print("selected_indices_diff_mask: ", selected_indices_diff_mask.shape[0])
            selected_indices = np.concatenate((selected_indices, selected_indices_diff_mask), axis=0)

        else:
            if fg_mask is not None:
                selected_indices = np.random.choice(np.nonzero(fg_mask.reshape(-1))[0], int(ray_bundle_size), replace=False)
            else:
                selected_indices = np.random.choice(ray_origins.shape[0], int(ray_bundle_size), replace=False)

        selected_indices = torch.from_numpy(selected_indices).long()

        ray_o = ray_origins[selected_indices].float().cuda()
        ray_d = ray_dirs[selected_indices].float().cuda()

        rgb = rgb[selected_indices].cuda()
        normal = normal[selected_indices].cuda()
        mask = mask[selected_indices].cuda()
        if nm_mask is not None:
            nm_mask = nm_mask[selected_indices].cuda()
        if depth is not None:
            depth = torch.from_numpy(depth).float().reshape(-1)[selected_indices].cuda()
        if depth_mask is not None:
            depth_mask = torch.from_numpy(depth_mask).float().reshape(-1)[selected_indices].cuda()
        if sm_mask is not None:
            sm_mask = sm_mask[selected_indices].cuda().float()

        normal = torch.nn.functional.normalize(normal, dim=-1)

        color_detach_geo = self.invis_loss_conf.get("color_detach_geo", False)
        if color_detach_geo and image_source != 'sdf':
            out = local_model.forward_multi_obj_rays_subset_all_sdf_detach_rgb_for_geometry_near_far(
                ray_o.cuda(), ray_d.cuda(), pose, subset_idxs, subset_idxs,
                near, far)
        else:
            out = local_model.forward_multi_obj_rays_subset_all_sdf_near_far(ray_o.cuda(), ray_d.cuda(), pose, subset_idxs, subset_idxs,
                                                                             near, far)

        rgb_pred = out['rgb_values'].reshape(-1, 3)
        normal_pred = out['normal_map'].reshape(-1, 3)
        mask_pred = out['opacity'].reshape(-1)
        depth_pred = out['depth_values'].reshape(-1)

        rgb_pred = rgb_pred * mask_pred.unsqueeze(-1) + (1 - mask_pred.unsqueeze(-1)) * (torch.ones_like(rgb_pred) * torch.from_numpy(bg_color).reshape(1, 3).cuda().float()).to(rgb_pred.device)

        lambda_mask = 25 * self.invis_loss_conf["lambda_mask"] if front else self.invis_loss_conf["lambda_mask"]
        lambda_nm_l1 = self.invis_loss_conf["lambda_nm_l1"]
        lambda_nm_cos = self.invis_loss_conf["lambda_nm_cos"]
        lambda_rgb = self.invis_loss_conf["lambda_rgb"]
        lambda_depth = self.invis_loss_conf["lambda_depth"]

        if image_source == "lama":
            lambda_rgb = self.invis_loss_conf.get('lambda_lama_rgb', lambda_rgb)
            lambda_nm_l1 = self.invis_loss_conf.get('lambda_lama_nm_l1', lambda_nm_l1)
            lambda_nm_cos = self.invis_loss_conf.get('lambda_lama_nm_cos', lambda_nm_cos)

        if 'lambda_mask' in gen_data_dict:
            lambda_mask = gen_data_dict.get('lambda_mask', lambda_mask)
            lambda_rgb = gen_data_dict.get('lambda_rgb', lambda_rgb)
            lambda_nm_l1 = gen_data_dict.get('lambda_nm_l1', lambda_nm_l1)
            lambda_nm_cos = gen_data_dict.get('lambda_nm_cos', lambda_nm_cos)
            lambda_depth = gen_data_dict.get('lambda_depth', lambda_depth)

            loss_opa = ((mask_pred - mask) ** 2).mean()
            loss_rgb = torch.nn.functional.l1_loss(rgb_pred, rgb).mean()
            loss_nm_l1 = torch.nn.functional.l1_loss(normal_pred, normal).mean()
            loss_nm_cos = 1 - torch.nn.functional.cosine_similarity(normal_pred, normal, dim=-1).mean()
            loss_depth = torch.nn.functional.l1_loss(depth_pred, depth).mean()

            novel_view_loss = lambda_mask * loss_opa + \
                              lambda_rgb * loss_rgb + \
                              lambda_nm_l1 * loss_nm_l1 + lambda_nm_cos * loss_nm_cos + \
                              lambda_depth * loss_depth

        # print("lambda_mask: ", lambda_mask, "lambda_rgb: ", lambda_rgb, "lambda_nm_l1: ", lambda_nm_l1, "lambda_nm_cos: ", lambda_nm_cos, "lambda_depth: ", lambda_depth)
        else:
            fg = mask > 0.
            nm_fg = nm_mask > 0. if nm_mask is not None else fg
            depth_fg = depth_mask > 0. if depth_mask is not None else fg

            # print("obj_idx: ", obj_idx, "data_idx: ", data_idx)

            loss_opa = torch.nn.functional.mse_loss(mask_pred, mask).mean()
            # loss_opa = ((mask_pred - mask)**2).mean()

            novel_view_loss = torch.tensor(0.0).cuda().float()

            if not torch.isnan(loss_opa):
                novel_view_loss = lambda_mask * loss_opa

            if torch.any(fg):
                loss_rgb = torch.nn.functional.l1_loss(rgb_pred[fg], rgb[fg]).mean()
                if not torch.isnan(loss_rgb):
                    novel_view_loss = novel_view_loss + lambda_rgb * loss_rgb
                if torch.any(nm_fg):
                    loss_nm_l1 = torch.nn.functional.l1_loss(normal_pred[nm_fg], normal[nm_fg]).mean()
                    loss_nm_cos = 1 - torch.nn.functional.cosine_similarity(normal_pred[nm_fg], normal[nm_fg], dim=-1).mean()
                    if not torch.isnan(loss_nm_l1) and not torch.isnan(loss_nm_cos):
                        novel_view_loss = novel_view_loss + lambda_nm_l1 * loss_nm_l1 + lambda_nm_cos * loss_nm_cos
                if torch.any(depth_fg) and depth is not None:
                    loss_depth = torch.nn.functional.l1_loss(depth_pred[depth_fg], depth[depth_fg]).mean()
                    if not torch.isnan(loss_depth):
                        novel_view_loss = novel_view_loss + lambda_depth * loss_depth



        return novel_view_loss * loss_lambda


    def calculate_background_recon_loss(self, bg_info, local_model):
        bg_views_dict = bg_info[np.random.choice(len(bg_info))]

        rgb = bg_views_dict["rgb"]
        normal = bg_views_dict["normal"]
        depth = bg_views_dict["depth"]
        mask = bg_views_dict["mask"]
        pose = bg_views_dict["pose"]
        intrinsics = bg_views_dict["intrinsics"]

        H, W = normal.shape[:2]
        fx, fy, cx, cy = intrinsics

        rgb = torch.from_numpy(rgb).float().reshape(-1, 3)
        normal = torch.from_numpy(normal).float().reshape(-1, 3)
        depth = torch.from_numpy(depth).float().reshape(-1, 1)

        pose = pose.cuda()

        ray_bundle_size = min(1024, np.count_nonzero(mask.reshape(-1)))

        selected_indices = np.random.choice(np.nonzero(mask.reshape(-1))[0].reshape(-1), ray_bundle_size, replace=False)
        selected_indices = torch.from_numpy(selected_indices).long()

        ray_origins, ray_dirs = get_camera_perspective_rays_world(fx, fy, cx, cy, H, W, pose.cpu().numpy())
        ray_o = torch.from_numpy(ray_origins.reshape(-1, 3))[selected_indices]
        ray_d = torch.from_numpy(ray_dirs.reshape(-1, 3))[selected_indices]

        rgb = rgb[selected_indices].cuda()
        normal = normal[selected_indices].cuda()
        depth = depth[selected_indices].cuda()

        out = local_model.forward_multi_obj_rays_subset_all_sdf(ray_o.cuda(), ray_d.cuda(), pose, [0], [0])
        rgb_pred = out['rgb_values'].reshape(-1, 3)
        normal_pred = out['normal_map'].reshape(-1, 3)
        depth_pred = out['depth_values'].reshape(-1, 1)

        lambda_nm_l1 = self.invis_loss_conf.get("bg_nm_l1", 10.0)
        lambda_nm_cos = self.invis_loss_conf.get("bg_nm_cos", 10.0)
        lambda_depth = self.invis_loss_conf.get("bg_depth", 0.0)
        lambda_rgb = 2.0

        loss_rgb = torch.nn.functional.l1_loss(rgb_pred, rgb).mean()
        loss_nm_l1 = torch.nn.functional.l1_loss(normal_pred, normal).mean()
        loss_nm_cos = 1 - torch.nn.functional.cosine_similarity(normal_pred, normal, dim=-1).mean()
        loss_depth = torch.nn.functional.l1_loss(depth_pred, depth).mean()

        view_loss = lambda_nm_l1 * loss_nm_l1 + lambda_nm_cos * loss_nm_cos + lambda_rgb * loss_rgb + lambda_depth * loss_depth

        return view_loss


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


    def get_obj_bbox(self, epoch):
        all_mesh_dict = self.get_all_meshes(epoch)

        all_mesh_bbox_dict = {}

        for obj_i in sorted(list(all_mesh_dict.keys())):
            obj_i_mesh = all_mesh_dict[obj_i]
            obj_i_verts_min = np.min(obj_i_mesh.vertices, axis=0).reshape(3)
            obj_i_verts_max = np.max(obj_i_mesh.vertices, axis=0).reshape(3)
            obj_i_bbox_center = (obj_i_verts_min + obj_i_verts_max) / 2
            obj_i_scale_xyz = obj_i_verts_max - obj_i_verts_min
            obj_i_bbox_scale = float(np.max(obj_i_scale_xyz))

            all_mesh_bbox_dict[obj_i] = {
                "center": obj_i_bbox_center,
                "scale": obj_i_bbox_scale,
                "scale_xyz": obj_i_scale_xyz,
            }

        return all_mesh_bbox_dict

    def generative_sampling(self):


        self.model.eval()

        num_objs = self.model.implicit_network.d_out
        all_mesh_dict = self.get_all_meshes(self.start_epoch)

        all_mesh_view_pruned_dict = self.get_all_meshes_view_pruned(all_mesh_dict)

        all_mesh_dict[0] = self.mesh_coarse_recon_dict[0]

        all_mesh_list = []
        sem_colors = np.random.rand(num_objs, 3) * 255

        save_view_vis_dir = os.path.join(self.plots_dir, "view_vis")
        save_view_weight_dir = os.path.join(self.plots_dir, "view_weight")
        os.makedirs(save_view_vis_dir, exist_ok=True)
        os.makedirs(save_view_weight_dir, exist_ok=True)

        save_rendering_dir = os.path.join(self.plots_dir, "rendering")
        os.makedirs(save_rendering_dir, exist_ok=True)
        save_lama_dir = os.path.join(self.plots_dir, "lama")
        os.makedirs(save_lama_dir, exist_ok=True)


        for obj_i in range(num_objs):
            all_mesh_list.append(all_mesh_dict[obj_i])

        eps_bg = 0.03
        resolution = self.invis_loss_conf.get('image_res', 256)
        nm_deviated_threshold = 1.05
        lama_inpaint_normal = True
        lama_dilate_iterations = 4
        stable_orientation_threshold = 8.

        mv_w3d, projs_w3d = make_wonder3D_cameras()

        vis_info = {}

        objs_seq_with_distance = [(obj_i, self.model.graph_node_dict[obj_i]['dist_to_root']) for obj_i in range(1, num_objs)]
        # sort objs_seq_with_distance by distance
        objs_seq_with_distance.sort(key=lambda x: x[1])
        objs_seq = [obj_i[0] for obj_i in objs_seq_with_distance]


        for obj_i in range(num_objs):
            self.train_dataset.graph_node_dict[obj_i]["brothers"] = []
            if obj_i == 0:
                continue


            brother_list = []
            parent_id = self.train_dataset.graph_node_dict[obj_i]['parent']
            parent_descs = self.train_dataset.graph_node_dict[parent_id]['desc']
            for desc in parent_descs:
                if desc == obj_i:
                    continue
                if self.train_dataset.graph_node_dict[desc]['parent'] != parent_id:
                    continue
                brother_list = brother_list + ([desc] + self.train_dataset.graph_node_dict[desc]['desc'])

            obj_i_center = self.original_bbox_dict[obj_i]['center']
            obj_i_scale = self.original_bbox_dict[obj_i]['scale_xyz'] * 0.6

            obj_i_x_min = obj_i_center[0] - obj_i_scale[0]
            obj_i_x_max = obj_i_center[0] + obj_i_scale[0]
            obj_i_y_min = obj_i_center[1] - obj_i_scale[1]
            obj_i_y_max = obj_i_center[1] + obj_i_scale[1]
            obj_i_z_min = obj_i_center[2] - obj_i_scale[2]
            obj_i_z_max = obj_i_center[2] + obj_i_scale[2]

            for brother in brother_list:
                brother_center = self.original_bbox_dict[brother]['center']
                brother_scale = self.original_bbox_dict[brother]['scale_xyz']

                brother_x_min = brother_center[0] - brother_scale[0]
                brother_x_max = brother_center[0] + brother_scale[0]
                brother_y_min = brother_center[1] - brother_scale[1]
                brother_y_max = brother_center[1] + brother_scale[1]
                brother_z_min = brother_center[2] - brother_scale[2]
                brother_z_max = brother_center[2] + brother_scale[2]

                # find possible intersection between brother bbox and obj_i bbox
                if not ((obj_i_x_min > brother_x_max or obj_i_x_max < brother_x_min) or
                        (obj_i_y_min > brother_y_max or obj_i_y_max < brother_y_min) or
                        (obj_i_z_min > brother_z_max or obj_i_z_max < brother_z_min)):
                    self.train_dataset.graph_node_dict[obj_i]["brothers"].append(brother)

            print("obj_i: ", obj_i, "brother_list: ", self.train_dataset.graph_node_dict[obj_i]["brothers"])
        failed_object_list = []
        for obj_i in tqdm(objs_seq):

            if os.path.exists(os.path.join(self.finetune_folder, "plots", f"coarse_recon_obj_{obj_i}.ply")):
                print("loading coarse recon mesh from finetune folder: ", obj_i)
                load_base_dir = os.path.join(self.finetune_folder, "plots")
                self.mesh_coarse_recon_dict[obj_i] = trimesh.exchange.load.load_mesh(
                    load_base_dir+f"/coarse_recon_obj_{obj_i}.ply")
                self.mesh_coarse_points_collisions_dict[obj_i] = pickle.load(
                    open(load_base_dir + f"/coarse_recon_obj_collision_pts_sdf_{obj_i}.pkl", 'rb'))
                print("self.mesh_coarse_points_collisions_dict[obj_i]: ", self.mesh_coarse_points_collisions_dict[obj_i].keys())
                load_base_dir = os.path.abspath(load_base_dir)
                ply_link_path = os.path.join(self.plots_dir, f'coarse_recon_obj_{obj_i}.ply')
                pkl_link_path = os.path.join(self.plots_dir, f'coarse_recon_obj_collision_pts_sdf_{obj_i}.pkl')
                if not os.path.exists(ply_link_path):
                    os.symlink(os.path.join(load_base_dir, f'coarse_recon_obj_{obj_i}.ply'), ply_link_path)
                if not os.path.exists(pkl_link_path):
                    os.symlink(os.path.join(load_base_dir, f'coarse_recon_obj_collision_pts_sdf_{obj_i}.pkl'), pkl_link_path)

                all_mesh_list[obj_i] = self.mesh_coarse_recon_dict[obj_i]
                all_mesh_list = self.instance_meshes_post_pruning_selected(self.start_epoch, input_meshes=all_mesh_list, selected_idx=obj_i)
                self.mesh_coarse_recon_dict[obj_i] = all_mesh_list[obj_i]

                all_mesh_dict[obj_i] = self.mesh_coarse_recon_dict[obj_i]


                continue

            coarse_mesh_list_for_phy_test = []
            obj_i_traverse_tree = obj_i
            while True:
                obj_i_traverse_tree = self.model.graph_node_dict[obj_i_traverse_tree]['parent']
                coarse_mesh_list_for_phy_test.append(self.mesh_coarse_recon_dict[obj_i_traverse_tree])
                if obj_i_traverse_tree == 0:
                    break
            coarse_mesh_list_for_phy_test = coarse_mesh_list_for_phy_test[::-1]

            torch.cuda.empty_cache()
            parent_idx = obj_i
            bg_idxs = [parent_idx]
            descs = self.model.graph_node_dict[parent_idx]["desc"]

            subset_idxs = bg_idxs + descs
            print("obj_i: ", obj_i, "subset_idxs: ", subset_idxs)

            view_image_dict_list = []

            scale_obj_i, shift_obj_i = get_scale_shift(all_mesh_view_pruned_dict[obj_i].vertices)
            scale_obj_i = float(scale_obj_i)
            shift_obj_i = torch.from_numpy(shift_obj_i).reshape(1, 3)

            all_meshes_reordered = []
            for subidx in subset_idxs:
                if subidx in all_mesh_dict:
                    all_meshes_reordered.append(all_mesh_view_pruned_dict[subidx])
                else:
                    all_meshes_reordered.append(
                        trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)), process=False))

            # lama

            if len(descs) > 0:
                vis_weight_dict = self.get_view_weights_of_subset_meshes_with_training_views_backface_discount_limited_phi(
                    all_mesh_view_pruned_dict, subset_idxs)
                view_weight_threshold = 0.75
                view_weights = vis_weight_dict["view_weights"]

                visualize_view_weights(view_weights,
                                       os.path.join(save_view_vis_dir, f"view_weights_{obj_i:0>2d}_f1_lama_before.png"))

                with open(os.path.join(save_view_weight_dir, f"view_weights_lama_{obj_i:0>2d}.pkl"), "wb") as f:
                    pickle.dump(vis_weight_dict, f)

                continous_azi_indices, _ = find_longest_continuous_azimuths(view_weights)
                num_pick_views = 5

                picked_azi_indices = [item for idx, item in enumerate(continous_azi_indices) if idx in np.linspace(
                    int(0.3 * (len(continous_azi_indices) - 1)),
                    int(0.7 * (len(continous_azi_indices) - 1)), num_pick_views).astype(np.int32)]

                picked_views = []
                for azi_idx in picked_azi_indices:
                    picked_azi_i = vis_weight_dict["azimuth"][azi_idx]
                    azi_i_alt_indices_over_threshold = \
                    np.nonzero(view_weights[azi_idx].reshape(-1) / view_weights.max() > view_weight_threshold)[0]
                    alt_idx = azi_i_alt_indices_over_threshold[
                        np.argsort(view_weights[azi_idx, azi_i_alt_indices_over_threshold])[
                            len(azi_i_alt_indices_over_threshold) // 2]]
                    picked_alt_i = vis_weight_dict["altitude"][alt_idx]
                    picked_views.append([picked_azi_i, picked_alt_i])

                visualize_view_weights_with_highlighted_azimuths(view_weights, picked_views, continous_azi_indices,
                                                                 picked_azi_indices,
                                                                 os.path.join(save_view_vis_dir,
                                                                              f"view_weights_lama_{obj_i:0>2d}_f3_info.png"))

                scale = vis_weight_dict["scale"]
                shift = vis_weight_dict["center"]

                scale_2, shift_2 = get_scale_shift(all_mesh_view_pruned_dict[parent_idx].vertices)

                for view_i, view in enumerate(picked_views):
                    theta_view = view[0]
                    phi_view = view[1]

                    if len(descs) > 0:
                        pose = build_camera_matrix_from_angles_and_locs_diff(theta_view, phi_view, scale, shift,
                                                                             shift_2)
                        theta_view, phi_view = get_theta_phi(pose[..., :3, 3].reshape(3).cpu().numpy(),
                                                             shift_2.reshape(3))
                    else:
                        pose = build_camera_matrix_from_angles_and_locs(theta_view, phi_view, scale, shift)
                    pose = pose.cuda()

                    H, W = resolution, resolution
                    if len(descs) > 0:
                        ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near=0.001, pose=pose.cpu().clone(),
                                                                           scale=scale_2)

                        cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=scale_2)
                    else:
                        ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near=0.001, pose=pose.cpu().clone(),
                                                                           scale=scale)

                        cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=scale)
                    cam_proj = torch.from_numpy(cam_proj).float().cuda()

                    mvp = cam_proj @ torch.inverse(pose.clone())

                    valid, _, _, mesh_sem = rasterize_mesh_list_front_face(all_meshes_reordered, mvp, self.glctx,
                                                                           (H, W), pose)
                    mesh_sem[~valid] = -1
                    mesh_sem = mesh_sem.cpu()

                    mesh_opa = torch.isin(mesh_sem, torch.arange(len(subset_idxs)))
                    mesh_desc_opa = torch.isin(mesh_sem, torch.arange(1, len(subset_idxs)))
                    mesh_self_opa = mesh_sem == 0

                    mesh_opa = mesh_opa.reshape(H, W).detach().cpu().numpy()
                    mesh_desc_opa = mesh_desc_opa.reshape(H, W).detach().cpu().numpy()
                    mesh_self_opa = mesh_self_opa.reshape(H, W).detach().cpu().numpy()


                    ray_origins = ray_origins.reshape(-1, 3)
                    ray_dirs = ray_dirs.reshape(-1, 3)

                    rgb_list = []
                    nm_list = []
                    depth_list = []

                    for ray_o, ray_d in zip(torch.split(ray_origins, 1024), torch.split(ray_dirs, 1024)):
                        out = self.model.forward_multi_obj_rays_subset_all_sdf(ray_o.cuda(), ray_d.cuda(), pose,
                                                                               subset_idxs, subset_idxs)

                        rgb_list.append(out['rgb_values'].detach().cpu())
                        nm_list.append(out['normal_map'].detach().cpu())
                        depth_list.append(out['depth_values'].detach().cpu())

                    rgb = torch.cat(rgb_list, dim=0)
                    nm = torch.cat(nm_list, dim=0)
                    depth = torch.cat(depth_list, dim=0)

                    rgb = rgb.reshape(H, W, 3).detach().cpu().numpy()
                    nm = nm.reshape(H, W, 3).detach().cpu().numpy()
                    depth = depth.reshape(H, W, 1).detach().cpu().numpy()

                    rgb_mesh_masked = rgb.copy()
                    rgb_mesh_masked[mesh_opa] = rgb_mesh_masked[mesh_opa] * 0.5 + np.array([1., 0., 0.]) * 0.5

                    bg_color = np.array([1., 1., 1.]).astype(np.float32).reshape(3)
                    rgb = np.clip(rgb, 0, 0.9)

                    rgb[~mesh_opa] = bg_color
                    im_label = f"{parent_idx:0>2d}_lama_view_{int(view_i):0>2d}_{int(theta_view):0>3d}_{int(phi_view):0>3d}"

                    rgb_path = os.path.join(save_rendering_dir, f"{im_label}_rgb.png")
                    rgb_raw_path = os.path.join(save_rendering_dir, f"{im_label}_rgb_raw.png")
                    nm_path = os.path.join(save_rendering_dir, f"{im_label}_nm.png")
                    depth_path = os.path.join(save_rendering_dir, f"{im_label}_depth.png")
                    mesh_opa_path = os.path.join(save_rendering_dir, f"{im_label}_mesh_opa.png")

                    Image.fromarray(
                        np.clip(np.concatenate([rgb, mesh_opa.astype(np.float32)[..., None]], axis=-1) * 255, 0,
                                255).astype(np.uint8)).save(rgb_path)
                    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(rgb_raw_path)
                    Image.fromarray(np.clip(
                        np.concatenate([((nm + 1.) * 0.5), mesh_opa.astype(np.float32)[..., None]], axis=-1) * 255, 0,
                        255).astype(np.uint8)).save(nm_path)
                    Image.fromarray(np.clip(mesh_opa * 255, 0, 255).astype(np.uint8)).save(mesh_opa_path)

                    # lama
                    rgb_to_inpaint = rgb.copy()
                    bg_region_lama = np.logical_or(~mesh_opa, mesh_desc_opa)

                    bg_color = np.array([1., 1., 1.]).astype(np.float32).reshape(3)
                    rgb_to_inpaint = np.clip(rgb_to_inpaint, 0, 0.9)
                    rgb_to_inpaint[bg_region_lama] = bg_color

                    rgb_to_inpaint_path = os.path.join(save_lama_dir, f"{im_label}_rgb_f1_to_inpaint.png")
                    rgb_to_inpaint = np.clip(rgb_to_inpaint, 0, 1)

                    Image.fromarray(np.clip(rgb_to_inpaint * 255, 0, 255).astype(np.uint8)).save(rgb_to_inpaint_path)
                    mesh_desc_opa = binary_dilation(mesh_desc_opa, iterations=lama_dilate_iterations)
                    rgb_inpainted = inpaint(self.lama_model, self.lama_predict_config, torch.from_numpy(rgb_to_inpaint),
                                            torch.from_numpy(mesh_desc_opa))
                    rgb_inpainted = rgb_inpainted.cpu().numpy()
                    rgb_inpainted_path = os.path.join(save_lama_dir, f"{im_label}_rgb_f2_inpainted.png")
                    Image.fromarray(np.clip(rgb_inpainted * 255, 0, 255).astype(np.uint8)).save(
                        rgb_inpainted_path)

                    nm_to_inpaint = nm.copy()
                    nm_to_inpaint = nm_to_inpaint * 0.5 + 0.5
                    nm_to_inpaint[bg_region_lama] = bg_color
                    nm_inpainted_from_lama_raw = inpaint(self.lama_model, self.lama_predict_config,
                                                         torch.from_numpy(nm_to_inpaint),
                                                         torch.from_numpy(mesh_desc_opa))
                    nm_inpainted_from_lama = nm_inpainted_from_lama_raw * 2 - 1
                    nm_inpainted_from_lama = nm_inpainted_from_lama.cpu().numpy()

                    gen_alpha_nm_lama = np.any(
                        np.abs(nm_inpainted_from_lama_raw.cpu().numpy() - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)
                    gen_alpha_nm_lama = np.logical_or(gen_alpha_nm_lama, mesh_self_opa)

                    depth_vis = np.concatenate([depth.copy()] * 3, axis=-1)
                    depth_to_inpaint = depth_vis
                    fg_region_lama = np.logical_not(bg_region_lama)
                    depth_min = np.min(depth_to_inpaint[fg_region_lama]) - 0.1
                    depth_max = np.max(depth_to_inpaint[fg_region_lama]) + 0.1
                    depth_to_inpaint = (depth_to_inpaint - depth_min) / (depth_max - depth_min)
                    depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
                    depth_to_inpaint[bg_region_lama] = bg_color
                    depth_inpainted_from_lama_raw = inpaint(self.lama_model, self.lama_predict_config,
                                                            torch.from_numpy(depth_to_inpaint),
                                                            torch.from_numpy(mesh_desc_opa))
                    depth_inpainted_from_lama = depth_inpainted_from_lama_raw.cpu().numpy()
                    gen_alpha_depth_lama = np.any(
                        np.abs(depth_inpainted_from_lama - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)
                    depth_inpainted_from_lama = depth_inpainted_from_lama.mean(axis=-1).reshape(H, W)
                    gen_alpha_depth_lama = np.logical_or(gen_alpha_depth_lama, mesh_self_opa)
                    depth_inpainted_from_lama_vis = depth_inpainted_from_lama.copy()
                    depth_inpainted_from_lama = depth_inpainted_from_lama * (depth_max - depth_min) + depth_min

                    normal_map_from_depth = get_normal_map_from_depth(depth_inpainted_from_lama, gen_alpha_depth_lama,
                                                                      scale_2)
                    normal_map_from_depth[np.logical_not(mesh_desc_opa)] = nm[np.logical_not(mesh_desc_opa)]

                    gen_color_lama = rgb_inpainted
                    gen_alpha_lama = np.any(np.abs(rgb_inpainted - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)

                    gen_alpha_lama = np.logical_or(gen_alpha_lama, mesh_self_opa)
                    gen_alpha_nm_lama = np.logical_and(gen_alpha_nm_lama, gen_alpha_lama)
                    gen_alpha_depth_lama = np.logical_and(gen_alpha_depth_lama, gen_alpha_lama)

                    # calculate deviate ratio

                    normal_map_from_depth = normal_map_from_depth / np.linalg.norm(normal_map_from_depth,
                                                                                   axis=-1, keepdims=True)
                    nm_inpainted_from_lama = nm_inpainted_from_lama / np.linalg.norm(nm_inpainted_from_lama,
                                                                                     axis=-1, keepdims=True)

                    lama_new_gen_region_mask = np.logical_and(gen_alpha_lama, mesh_desc_opa)
                    nm_inpainted_from_lama_new_gen = nm_inpainted_from_lama[lama_new_gen_region_mask].reshape(-1, 3)
                    normal_map_from_depth_new_gen = normal_map_from_depth[lama_new_gen_region_mask].reshape(-1, 3)

                    if len(nm_inpainted_from_lama_new_gen) > 0:
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 30 degrees
                        deviated_nm_ratio = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen,
                                   axis=-1) < 0.866) / len(
                            nm_inpainted_from_lama_new_gen)
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 45 degrees
                        deviated_nm_ratio_2 = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen,
                                   axis=-1) < 0.707) / len(
                            nm_inpainted_from_lama_new_gen)
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 60 degrees
                        deviated_nm_ratio_3 = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen,
                                   axis=-1) < 0.5) / len(
                            nm_inpainted_from_lama_new_gen)
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 90 degrees
                        deviated_nm_ratio_4 = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen,
                                   axis=-1) < 0.0) / len(
                            nm_inpainted_from_lama_new_gen)

                        deviated = deviated_nm_ratio > 0.4 or deviated_nm_ratio_2 > 0.3 or deviated_nm_ratio_3 > 0.2 or deviated_nm_ratio_4 > 0.1
                    else:
                        deviated = False
                    gen_lama_im_label = f"{parent_idx:0>2d}_lama_view_{int(view_i):0>2d}_gen_lama_view_{int(theta_view):0>3d}_{int(phi_view):0>3d}"

                    inpaint_mask_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_inpaint_mask.png")
                    gen_lama_rgb_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_rgb.png")
                    gen_lama_nm_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_nm.png")
                    gen_lama_nm_raw_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_nm_raw.png")
                    gen_lama_nm_from_lama_path = os.path.join(save_rendering_dir,
                                                              f"{gen_lama_im_label}_nm_lama_deviated_{deviated}.png")
                    gen_lama_depth_from_lama_path = os.path.join(save_rendering_dir,
                                                                 f"{gen_lama_im_label}_depth_lama.png")
                    gen_normal_map_from_depth_path = os.path.join(save_rendering_dir,
                                                                  f"{gen_lama_im_label}_normal_map_from_depth.png")
                    gen_lama_rgb_fg_mask_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_rgb_fg_mask.png")

                    gen_color_lama_fg_mask = gen_color_lama.copy()
                    gen_color_lama_fg_mask[mesh_desc_opa] = gen_color_lama_fg_mask[
                                                                    mesh_desc_opa] * 0.5 + np.array(
                        [1., 0., 0.]) * 0.5

                    Image.fromarray(
                        np.clip(mesh_desc_opa.astype(np.float32) * 255, 0, 255).astype(np.uint8)).save(inpaint_mask_path)

                    Image.fromarray(
                        np.clip(np.concatenate([gen_color_lama, gen_alpha_lama.astype(np.float32)[..., None]],
                                               axis=-1) * 255, 0,
                                255).astype(np.uint8)).save(gen_lama_rgb_path)

                    Image.fromarray(
                        np.clip(np.concatenate([gen_color_lama_fg_mask, gen_alpha_lama.astype(np.float32)[..., None]],
                                               axis=-1) * 255, 0,
                                255).astype(np.uint8)).save(gen_lama_rgb_fg_mask_path)

                    Image.fromarray(np.clip(
                        np.concatenate(
                            [((nm_inpainted_from_lama + 1.) * 0.5), gen_alpha_nm_lama.astype(np.float32)[..., None]],
                            axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_lama_nm_from_lama_path)

                    Image.fromarray(np.clip(
                        np.concatenate(
                            [np.concatenate([depth_inpainted_from_lama_vis[..., None]] * 3, axis=-1),
                             gen_alpha_depth_lama.astype(np.float32)[..., None]],
                            axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_lama_depth_from_lama_path)

                    Image.fromarray(
                        np.clip(np.concatenate([depth_vis.astype(np.float32),
                                            mesh_opa.astype(np.float32)[..., None]], axis=-1) * 255, 0, 255).astype(np.uint8)).save(depth_path)

                    Image.fromarray(np.clip(
                        np.concatenate(
                            [((normal_map_from_depth + 1.) * 0.5),
                             gen_alpha_depth_lama.astype(np.float32)[..., None]],
                            axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_normal_map_from_depth_path)

                    view_image_dict_list.append({
                        'rgb': gen_color_lama,
                        'normal': normal_map_from_depth if deviated else nm_inpainted_from_lama,
                        'depth': depth_inpainted_from_lama,
                        'mask': gen_alpha_lama,
                        'nm_mask': gen_alpha_nm_lama,
                        'depth_mask': gen_alpha_depth_lama,
                        "pose": pose.cpu(),
                        "scale": scale_2,
                        "obj_idxs": [parent_idx],
                        "lambda": 1.0,
                        "front": True,
                        "bg_color": bg_color,
                        'sm_mask': mesh_desc_opa,
                        'source': "lama",
                        'offset': shift_2,
                        "fg_mask": mesh_desc_opa,
                    })


            # original
            all_stable_view_image_dict_list = {}
            all_stable_view_coarse_sampling = {}

            vis_weight_dict = self.get_view_weights_of_subset_meshes_with_training_views_backface_discount_limited_phi_strict_bbox(all_mesh_view_pruned_dict, subset_idxs)
            view_weight_threshold = 0.75
            view_weights = vis_weight_dict["view_weights"]

            visualize_view_weights(view_weights, os.path.join(save_view_vis_dir, f"view_weights_{obj_i:0>2d}_f1_before.png"))

            with open(os.path.join(save_view_weight_dir, f"view_weights_{obj_i:0>2d}.pkl"), "wb") as f:
                pickle.dump(vis_weight_dict, f)

            best_azi, best_alt, best_uniformity, best_new_weights = find_best_additional_view(view_weights)

            visualize_view_weights(best_new_weights, os.path.join(save_view_vis_dir, f"view_weights_{obj_i:0>2d}_f2_after.png"))

            should_add, metrics = evaluate_view_addition(view_weights, best_new_weights)
            print("obj_i: ", obj_i, "for wonder3d: ", should_add, "uniformity_improvement: ",
                  metrics["uniformity_improvement"])

            full_view = (not should_add and len(descs) == 0)

            continous_azi_indices, _ = find_longest_continuous_azimuths(view_weights)
            num_pick_views = 3

            if full_view:
                picked_azi_indices = [item for idx, item in enumerate(continous_azi_indices) if idx in np.linspace(
                    int(0. * (len(continous_azi_indices) - 1)),
                    int(1. * (len(continous_azi_indices) - 1)), num_pick_views).astype(np.int32)]
            elif not should_add:
                picked_azi_indices = [item for idx, item in enumerate(continous_azi_indices) if idx in np.linspace(
                    int(0.3 * (len(continous_azi_indices) - 1)),
                    int(0.7 * (len(continous_azi_indices) - 1)), num_pick_views).astype(np.int32)]
            else:
                if num_pick_views >= len(continous_azi_indices):
                    picked_azi_indices = continous_azi_indices
                else:
                    picked_azi_indices = [item for idx, item in enumerate(continous_azi_indices) if idx in
                                      np.arange(num_pick_views) + (len(continous_azi_indices) - num_pick_views) // 2]

            picked_views = []
            for azi_idx in picked_azi_indices:
                picked_azi_i = vis_weight_dict["azimuth"][azi_idx]
                azi_i_alt_indices_over_threshold = np.nonzero(view_weights[azi_idx].reshape(-1) / view_weights.max() > view_weight_threshold)[0]
                alt_idx = azi_i_alt_indices_over_threshold[np.argsort(view_weights[azi_idx, azi_i_alt_indices_over_threshold])[len(azi_i_alt_indices_over_threshold) // 2]]
                picked_alt_i = vis_weight_dict["altitude"][alt_idx]
                picked_views.append([picked_azi_i, picked_alt_i])

            visualize_view_weights_with_highlighted_azimuths(view_weights, picked_views, continous_azi_indices, picked_azi_indices,
                                                             os.path.join(save_view_vis_dir, f"view_weights_{obj_i:0>2d}_f3_info.png"))

            scale = vis_weight_dict["scale"]
            shift = vis_weight_dict["center"]


            if len(descs) > 0:
                scale_2, shift_2 = get_scale_shift(all_mesh_view_pruned_dict[parent_idx].vertices)

            for view_i, view in enumerate(picked_views):

                view_list_i = []
                
                view_image_dict_list_view_i = []

                theta_view = view[0]
                phi_view = view[1]

                if len(descs) > 0:
                    pose = build_camera_matrix_from_angles_and_locs_diff(theta_view, phi_view, scale, shift, shift_2)
                    theta_view, phi_view = get_theta_phi(pose[..., :3, 3].reshape(3).cpu().numpy(), shift_2.reshape(3))
                else:
                    pose = build_camera_matrix_from_angles_and_locs(theta_view, phi_view, scale, shift)
                pose = pose.cuda()

                theta_view_rad = np.deg2rad(theta_view)
                phi_view_rad = np.deg2rad(phi_view)

                R_from_w3d = np.array([
                    [-np.sin(theta_view_rad), -np.cos(phi_view_rad) * np.cos(theta_view_rad), np.sin(phi_view_rad) * np.cos(theta_view_rad)],
                    [np.cos(theta_view_rad), -np.cos(phi_view_rad) * np.sin(theta_view_rad), np.sin(phi_view_rad) * np.sin(theta_view_rad)],
                    [0, np.sin(phi_view_rad), np.cos(phi_view_rad)]
                ])

                R_from_w3d = torch.from_numpy(R_from_w3d).cuda().float()

                H, W = resolution, resolution
                if len(descs) > 0:
                    ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near=0.001, pose=pose.cpu().clone(),
                                                                       scale=scale_2)

                    cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=scale_2)
                else:
                    ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near=0.001, pose=pose.cpu().clone(),
                                                                       scale=scale)

                    cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=scale)
                cam_proj = torch.from_numpy(cam_proj).float().cuda()

                mvp = cam_proj @ torch.inverse(pose.clone())

                valid, _, _, mesh_sem = rasterize_mesh_list_front_face(all_meshes_reordered, mvp, self.glctx, (H, W), pose)
                mesh_sem[~valid] = -1
                mesh_sem = mesh_sem.cpu()

                mesh_opa = torch.isin(mesh_sem, torch.arange(len(subset_idxs)))
                mesh_desc_opa = torch.isin(mesh_sem, torch.arange(1, len(subset_idxs)))
                mesh_self_opa = mesh_sem == 0

                mesh_opa = mesh_opa.reshape(H, W).detach().cpu().numpy()
                mesh_desc_opa = mesh_desc_opa.reshape(H, W).detach().cpu().numpy()
                mesh_self_opa = mesh_self_opa.reshape(H, W).detach().cpu().numpy()

                ray_origins = ray_origins.reshape(-1, 3)
                ray_dirs = ray_dirs.reshape(-1, 3)

                rgb_list = []
                nm_list = []
                depth_list = []

                for ray_o, ray_d in zip(torch.split(ray_origins, 1024), torch.split(ray_dirs, 1024)):
                    out = self.model.forward_multi_obj_rays_subset_all_sdf(ray_o.cuda(), ray_d.cuda(), pose, subset_idxs, subset_idxs)
                    # out = self.model.forward_multi_obj_rays_subset_all_sdf(ray_o.cuda(), ray_d.cuda(), pose, bg_idxs, bg_idxs)

                    rgb_list.append(out['rgb_values'].detach().cpu())
                    nm_list.append(out['normal_map'].detach().cpu())
                    depth_list.append(out['depth_values'].detach().cpu())

                rgb = torch.cat(rgb_list, dim=0)
                nm = torch.cat(nm_list, dim=0)
                depth = torch.cat(depth_list, dim=0)

                rgb = rgb.reshape(H, W, 3).detach().cpu().numpy()
                nm = nm.reshape(H, W, 3).detach().cpu().numpy()
                depth = depth.reshape(H, W, 1).detach().cpu().numpy()



                rgb_mesh_masked = rgb.copy()
                rgb_mesh_masked[mesh_opa] = rgb_mesh_masked[mesh_opa] * 0.5 + np.array([1., 0., 0.]) * 0.5

                bg_color = np.array([1., 1., 1.]).astype(np.float32).reshape(3)
                rgb = np.clip(rgb, 0, 0.9)

                rgb[~mesh_opa] = bg_color
                im_label = f"{parent_idx:0>2d}_view_{int(view_i):0>2d}_{int(theta_view):0>3d}_{int(phi_view):0>3d}"

                rgb_path = os.path.join(save_rendering_dir, f"{im_label}_rgb.png")
                rgb_raw_path = os.path.join(save_rendering_dir, f"{im_label}_rgb_raw.png")
                nm_path = os.path.join(save_rendering_dir, f"{im_label}_nm.png")
                depth_path = os.path.join(save_rendering_dir, f"{im_label}_depth.png")
                mesh_opa_path = os.path.join(save_rendering_dir, f"{im_label}_mesh_opa.png")

                Image.fromarray(np.clip(np.concatenate([rgb, mesh_opa.astype(np.float32)[..., None]], axis=-1) * 255, 0, 255).astype(np.uint8)).save(rgb_path)
                Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(rgb_raw_path)
                Image.fromarray(np.clip(np.concatenate([((nm + 1.) * 0.5), mesh_opa.astype(np.float32)[..., None]], axis=-1) * 255, 0, 255).astype(np.uint8)).save(nm_path)
                Image.fromarray(np.clip(mesh_opa * 255, 0, 255).astype(np.uint8)).save(mesh_opa_path)

                if full_view:
                    view_image_dict = {
                        'rgb': rgb,
                        'normal': nm,
                        'mask': mesh_opa,
                        "pose": pose.cpu(),
                        "lambda": 1.0,
                        "scale": scale,
                        "obj_idxs": subset_idxs,
                        "front": True,
                        "bg_color": bg_color,
                        'source': "sdf",
                        'offset': shift,
                    }
                    view_image_dict_list_view_i.append(view_image_dict)
                    view_list_i.append({
                        'rgb': rgb,
                        'normal': nm,
                        'mask': mesh_opa,
                        "pose": pose.cpu(),
                        "scale": scale,
                        "mv": mv_w3d[0],
                        "proj": projs_w3d
                    })
                elif len(descs) == 0:
                    view_image_dict = {
                        'rgb': rgb,
                        'normal': nm,
                        'mask': mesh_opa,
                        "pose": pose.cpu(),
                        "lambda": 1.0,
                        "scale": scale,
                        "obj_idxs": subset_idxs,
                        "front": True,
                        "bg_color": bg_color,
                        'source': "sdf",
                        'offset': shift,
                    }
                    view_image_dict_list_view_i.append(view_image_dict)
                    view_list_i.append({
                        'rgb': rgb,
                        'normal': nm,
                        'mask': mesh_opa,
                        "pose": pose.cpu(),
                        "scale": scale,
                        "mv": mv_w3d[0],
                        "proj": projs_w3d
                    })
                else:

                    # lama
                    rgb_to_inpaint = rgb.copy()
                    bg_region_lama = np.logical_or(~mesh_opa, mesh_desc_opa)
                    bg_color = np.array([1., 1., 1.]).astype(np.float32).reshape(3)
                    rgb_to_inpaint = np.clip(rgb_to_inpaint, 0, 0.9)
                    rgb_to_inpaint[bg_region_lama] = bg_color

                    rgb_to_inpaint_path = os.path.join(save_lama_dir, f"{im_label}_rgb_f1_to_inpaint.png")
                    rgb_to_inpaint = np.clip(rgb_to_inpaint, 0, 1)

                    Image.fromarray(np.clip(rgb_to_inpaint * 255, 0, 255).astype(np.uint8)).save(rgb_to_inpaint_path)
                    mesh_desc_opa = binary_dilation(mesh_desc_opa, iterations=lama_dilate_iterations)
                    rgb_inpainted = inpaint(self.lama_model, self.lama_predict_config, torch.from_numpy(rgb_to_inpaint),
                                            torch.from_numpy(mesh_desc_opa))
                    rgb_inpainted = rgb_inpainted.cpu().numpy()
                    rgb_inpainted_path = os.path.join(save_lama_dir, f"{im_label}_rgb_f2_inpainted.png")
                    Image.fromarray(np.clip(rgb_inpainted * 255, 0, 255).astype(np.uint8)).save(
                        rgb_inpainted_path)

                    nm_to_inpaint = nm.copy()
                    nm_to_inpaint = nm_to_inpaint * 0.5 + 0.5
                    nm_to_inpaint[bg_region_lama] = bg_color
                    nm_inpainted_from_lama_raw = inpaint(self.lama_model, self.lama_predict_config,
                                                     torch.from_numpy(nm_to_inpaint),
                                                     torch.from_numpy(mesh_desc_opa))
                    nm_inpainted_from_lama = nm_inpainted_from_lama_raw * 2 - 1
                    nm_inpainted_from_lama = nm_inpainted_from_lama.cpu().numpy()

                    gen_alpha_nm_lama = np.any(np.abs(nm_inpainted_from_lama_raw.cpu().numpy() - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)
                    gen_alpha_nm_lama = np.logical_or(gen_alpha_nm_lama, mesh_self_opa)

                    depth_vis = np.concatenate([depth.copy()] * 3, axis=-1)
                    depth_to_inpaint = depth_vis
                    fg_region_lama = np.logical_not(bg_region_lama)
                    depth_min = np.min(depth_to_inpaint[fg_region_lama]) - 0.1
                    depth_max = np.max(depth_to_inpaint[fg_region_lama]) + 0.1
                    depth_to_inpaint = (depth_to_inpaint - depth_min) / (depth_max - depth_min)
                    depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)
                    depth_to_inpaint[bg_region_lama] = bg_color
                    depth_inpainted_from_lama_raw = inpaint(self.lama_model, self.lama_predict_config,
                                                            torch.from_numpy(depth_to_inpaint),
                                                            torch.from_numpy(mesh_desc_opa))
                    depth_inpainted_from_lama = depth_inpainted_from_lama_raw.cpu().numpy()
                    gen_alpha_depth_lama = np.any(
                        np.abs(depth_inpainted_from_lama - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)
                    depth_inpainted_from_lama = depth_inpainted_from_lama.mean(axis=-1).reshape(H, W)
                    gen_alpha_depth_lama = np.logical_or(gen_alpha_depth_lama, mesh_self_opa)
                    depth_inpainted_from_lama_vis = depth_inpainted_from_lama.copy()
                    depth_inpainted_from_lama = depth_inpainted_from_lama * (depth_max - depth_min) + depth_min

                    normal_map_from_depth = get_normal_map_from_depth(depth_inpainted_from_lama, gen_alpha_depth_lama,
                                                                      scale_2)
                    normal_map_from_depth[np.logical_not(mesh_desc_opa)] = nm[np.logical_not(mesh_desc_opa)]

                    gen_color_lama = rgb_inpainted
                    gen_alpha_lama = np.any(np.abs(rgb_inpainted - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)

                    gen_alpha_lama = np.logical_or(gen_alpha_lama, mesh_self_opa)
                    gen_alpha_nm_lama = np.logical_and(gen_alpha_nm_lama, gen_alpha_lama)
                    gen_alpha_depth_lama = np.logical_and(gen_alpha_depth_lama, gen_alpha_lama)

                    # calculate deviate ratio

                    normal_map_from_depth = normal_map_from_depth / np.linalg.norm(normal_map_from_depth,
                                                                                   axis=-1, keepdims=True)
                    nm_inpainted_from_lama = nm_inpainted_from_lama / np.linalg.norm(nm_inpainted_from_lama,
                                                                                     axis=-1, keepdims=True)

                    lama_new_gen_region_mask = np.logical_and(gen_alpha_lama, mesh_desc_opa)
                    nm_inpainted_from_lama_new_gen = nm_inpainted_from_lama[lama_new_gen_region_mask].reshape(-1, 3)
                    normal_map_from_depth_new_gen = normal_map_from_depth[lama_new_gen_region_mask].reshape(-1, 3)

                    if len(nm_inpainted_from_lama_new_gen) > 0:
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 30 degrees
                        deviated_nm_ratio = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen, axis=-1) < 0.866) / len(
                            nm_inpainted_from_lama_new_gen)
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 45 degrees
                        deviated_nm_ratio_2 = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen, axis=-1) < 0.707) / len(
                            nm_inpainted_from_lama_new_gen)
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 60 degrees
                        deviated_nm_ratio_3 = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen, axis=-1) < 0.5) / len(
                            nm_inpainted_from_lama_new_gen)
                        # calculate the deviation ratio where the angle between the normal map from depth and the normal map from lama is larger than 90 degrees
                        deviated_nm_ratio_4 = np.count_nonzero(
                            np.sum(normal_map_from_depth_new_gen * nm_inpainted_from_lama_new_gen, axis=-1) < 0.0) / len(
                            nm_inpainted_from_lama_new_gen)

                        deviated = deviated_nm_ratio > 0.4 or deviated_nm_ratio_2 > 0.3 or deviated_nm_ratio_3 > 0.2 or deviated_nm_ratio_4 > 0.1
                    else:
                        deviated = False
                    gen_lama_im_label = f"{parent_idx:0>2d}_view_{int(view_i):0>2d}_gen_lama_view_{int(theta_view):0>3d}_{int(phi_view):0>3d}"

                    inpaint_mask_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_inpaint_mask.png")
                    gen_lama_rgb_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_rgb.png")
                    gen_lama_nm_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_nm.png")
                    gen_lama_nm_raw_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_nm_raw.png")
                    gen_lama_nm_from_lama_path = os.path.join(save_rendering_dir, f"{gen_lama_im_label}_nm_lama_deviated_{deviated}.png")
                    gen_lama_depth_from_lama_path = os.path.join(save_rendering_dir,
                                                                 f"{gen_lama_im_label}_depth_lama.png")
                    gen_normal_map_from_depth_path = os.path.join(save_rendering_dir,
                                                                  f"{gen_lama_im_label}_normal_map_from_depth.png")

                    Image.fromarray(
                        np.clip(mesh_desc_opa.astype(np.float32) * 255, 0, 255).astype(np.uint8)).save(
                        inpaint_mask_path)

                    Image.fromarray(
                        np.clip(np.concatenate([gen_color_lama, gen_alpha_lama.astype(np.float32)[..., None]], axis=-1) * 255, 0,
                                255).astype(np.uint8)).save(gen_lama_rgb_path)
                                
                    Image.fromarray(np.clip(
                        np.concatenate(
                            [((nm_inpainted_from_lama + 1.) * 0.5), gen_alpha_nm_lama.astype(np.float32)[..., None]],
                            axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_lama_nm_from_lama_path)

                    Image.fromarray(np.clip(
                        np.concatenate(
                            [np.concatenate([depth_inpainted_from_lama_vis[..., None]] * 3, axis=-1),
                             gen_alpha_depth_lama.astype(np.float32)[..., None]],
                            axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_lama_depth_from_lama_path)

                    Image.fromarray(
                        np.clip(np.concatenate([depth_vis.astype(np.float32),
                                                mesh_opa.astype(np.float32)[..., None]], axis=-1) * 255, 0, 255).astype(
                            np.uint8)).save(depth_path)

                    Image.fromarray(np.clip(
                        np.concatenate(
                            [((normal_map_from_depth + 1.) * 0.5),
                             gen_alpha_depth_lama.astype(np.float32)[..., None]],
                            axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_normal_map_from_depth_path)

                    view_image_dict = {
                        'rgb': gen_color_lama,
                        'normal': normal_map_from_depth if deviated else nm_inpainted_from_lama,
                        'depth': depth_inpainted_from_lama,
                        'mask': gen_alpha_lama,
                        'nm_mask': gen_alpha_nm_lama,
                        'depth_mask': gen_alpha_depth_lama,
                        "pose": pose.cpu(),
                        "scale": scale_2,
                        "obj_idxs": [parent_idx],
                        "lambda": 1.0,
                        "front": True,
                        "bg_color": bg_color,
                        'sm_mask': mesh_desc_opa,
                        'source': "lama",
                        'offset': shift_2,
                    }
                    view_image_dict_list_view_i.append(view_image_dict)

                    view_list_i.append({
                        'rgb': gen_color_lama,
                        'normal': normal_map_from_depth if deviated else nm_inpainted_from_lama,
                        'mask': gen_alpha_lama,
                        "pose": pose.cpu(),
                        "scale": scale_2,
                        "mv": mv_w3d[0],
                        "proj": projs_w3d
                    })

                    rgb = gen_color_lama
                    mesh_opa = gen_alpha_lama

                if not should_add:
                    view_image_dict_list = view_image_dict_list + view_image_dict_list_view_i
                    continue
                aug_select_idxs = []
                aug_view_delta_thetas = [45, 90., 180., 270., 315.]
                aug_view_phis = [float(np.arccos(1/np.sqrt(2) * np.cos(phi_view*np.pi/180.)) * 180. / np.pi),
                                 90., float(180.-phi_view), 90.,
                                 float(np.arccos(1/np.sqrt(2) * np.cos(phi_view*np.pi/180.)) * 180. / np.pi)]
                print("aug_view_delta_thetas: ", aug_view_delta_thetas)
                print("aug_view_phis: ", aug_view_phis)
                for aug_view_i, aug_view_delta_theta in enumerate(aug_view_delta_thetas):
                    delta_theta_to_invis = abs(theta_view + aug_view_delta_theta - best_azi) % 360
                    if delta_theta_to_invis > 180.:
                        delta_theta_to_invis = delta_theta_to_invis - 360.
                    if abs(delta_theta_to_invis) < 60:
                        aug_select_idxs.append(aug_view_i)
                print("aug_select_idxs: ", aug_select_idxs)

                if len(aug_select_idxs) > 0:

                    # wonder3d
                    rgb_to_gen = rgb.copy()
                    rgb_to_gen, mesh_opa = smooth_rgb_image(rgb_to_gen, mesh_opa)
                    mesh_opa = mesh_opa.copy() > 0
                    rgb_to_gen[~mesh_opa] = bg_color
                    rgb_to_gen = np.concatenate([rgb_to_gen, mesh_opa[..., None]], axis=-1)

                    rgb_to_gen_path = os.path.join(save_rendering_dir, f"{im_label}_rgb_to_gen.png")
                    Image.fromarray(np.clip(rgb_to_gen * 255, 0, 255).astype(np.uint8)).save(rgb_to_gen_path)

                    stable_view = False
                    for seed in range(42, 45):
                        view_image_dict_list_view_i_seed_i = []
                        view_list_i_seed_i = []

                        gen_normals, gen_colors, gen_alphas = wonder3d_generation(rgb_to_gen, self.mv_pipeline, self.config_mv, self.upsampler, sr_front_with_upsampler, seed=seed)

                        gen_normals = gen_normals[1:]
                        gen_colors = gen_colors[1:]
                        gen_alphas = gen_alphas[1:]

                        for aug_view_i in range(5):

                            gen_normal = gen_normals[aug_view_i]
                            gen_color = gen_colors[aug_view_i]
                            gen_alpha = np.any(np.abs(gen_color - bg_color.reshape(1, 1, 3)) > eps_bg, axis=-1)

                            theta_gen_view = (aug_view_delta_thetas[aug_view_i] + theta_view) % 360
                            phi_gen_view = aug_view_phis[aug_view_i]

                            if len(descs) > 0:
                                gen_view_pose = build_camera_matrix_from_angles_and_locs(theta_gen_view, phi_gen_view, scale, shift_2)
                            else:
                                gen_view_pose = build_camera_matrix_from_angles_and_locs(theta_gen_view, phi_gen_view, scale, shift)

                            gen_im_label = f"{parent_idx:0>2d}_view_{int(view_i):0>2d}_seed_{seed}_gen_view_{aug_view_i}_{int(theta_gen_view):0>3d}_{int(phi_gen_view):0>3d}"

                            gen_rgb_path = os.path.join(save_rendering_dir, f"{gen_im_label}_rgb.png")
                            gen_rgb_raw_path = os.path.join(save_rendering_dir, f"{gen_im_label}_rgb_raw.png")
                            gen_nm_path = os.path.join(save_rendering_dir, f"{gen_im_label}_nm.png")

                            Image.fromarray(np.clip(np.concatenate([gen_color, gen_alpha.astype(np.float32)[..., None]], axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_rgb_path)
                            Image.fromarray(np.clip(gen_color * 255, 0, 255).astype(np.uint8)).save(gen_rgb_raw_path)
                            Image.fromarray(np.clip(np.concatenate([((gen_normal + 1.) * 0.5), gen_alpha.astype(np.float32)[..., None]], axis=-1) * 255, 0, 255).astype(np.uint8)).save(gen_nm_path)

                            if len(descs) == 0:
                                view_image_dict = {
                                    'rgb': gen_color,
                                    'normal': gen_normal,
                                    'mask': gen_alpha,
                                    "pose": gen_view_pose.cpu(),
                                    "lambda": 0.2,
                                    "scale": scale,
                                    "shift": shift,
                                    "obj_idxs": subset_idxs,
                                    "front": False,
                                    "bg_color": bg_color,
                                    'sm_mask': gen_alpha,
                                    'source': "wonder3d",
                                }
                                if aug_view_i in aug_select_idxs:
                                    view_image_dict_list_view_i_seed_i.append(view_image_dict)
                                view_list_i_seed_i.append({
                                    'rgb': gen_color,
                                    'normal': gen_normal,
                                    'mask': gen_alpha,
                                    "pose": gen_view_pose.cpu(),
                                    "scale": scale,
                                    "mv": mv_w3d[aug_view_i+1],
                                    "proj": projs_w3d
                                })
                            else:
                                view_image_dict = {
                                    'rgb': gen_color,
                                    'normal': gen_normal,
                                    'mask': gen_alpha,
                                    "pose": gen_view_pose.cpu(),
                                    "lambda": 0.2,
                                    "scale": scale_2,
                                    "shift": shift,
                                    "obj_idxs": [parent_idx],
                                    "front": False,
                                    "bg_color": bg_color,
                                    'sm_mask': gen_alpha,
                                    'source': "wonder3d",
                                }
                                if aug_view_i in aug_select_idxs:
                                    view_image_dict_list_view_i_seed_i.append(view_image_dict)
                                view_list_i_seed_i.append({
                                    'rgb': gen_color,
                                    'normal': gen_normal,
                                    'mask': gen_alpha,
                                    "pose": gen_view_pose.cpu(),
                                    "scale": scale_2,
                                    "mv": mv_w3d[aug_view_i+1],
                                    "proj": projs_w3d
                                })

                        os.makedirs(os.path.join(self.plots_dir, "coarse_recon"), exist_ok=True)

                        try:
                            mesh_recon_view_i_seed_i, texture_dict = coarse_recon(view_list_i+view_list_i_seed_i, scale_obj_i, shift_obj_i, R_from_w3d, debug_dir=os.path.join(self.plots_dir, "coarse_recon"))
                        except:
                            print("coarse_recon failed")
                            mesh_recon_view_i_seed_i = None

                        if mesh_recon_view_i_seed_i is None:
                            print("mesh_recon_view_i_seed_i is None")
                            view_image_dict_list_view_i = view_image_dict_list_view_i + view_image_dict_list_view_i_seed_i
                            all_stable_view_image_dict_list[view_i] = view_image_dict_list_view_i
                            break

                        trimesh.exchange.export.export_mesh(mesh_recon_view_i_seed_i,
                                                            os.path.join(self.plots_dir,
                                                                         f"{parent_idx:0>2d}_view_{int(view_i):0>2d}_seed_{seed}.ply"))

                        coarse_mesh_list_for_phy_test_view_i_seed_i = coarse_mesh_list_for_phy_test + [mesh_recon_view_i_seed_i]
                        relative_orientation = sim_validation(coarse_mesh_list_for_phy_test_view_i_seed_i)
                        print("view_i: ", view_i, "seed: ", seed, "relative_orientation: ", relative_orientation)
                        if relative_orientation < stable_orientation_threshold:
                            view_image_dict_list_view_i = view_image_dict_list_view_i + view_image_dict_list_view_i_seed_i
                            if len(descs) > 0:
                                proj_scale = scale_2
                                pose_scale = scale
                                pose_shift = shift_2
                            else:
                                proj_scale = scale
                                pose_scale = scale
                                pose_shift = shift

                            
                            sampled_views_params = copy.deepcopy([mesh_recon_view_i_seed_i, texture_dict, all_mesh_dict,
                                                                                    obj_i, best_azi, pose_scale, pose_shift, proj_scale])

                            stable_view = True
                            break

                    if stable_view:
                        # view_image_dict_list = view_image_dict_list + view_image_dict_list_view_i
                        all_stable_view_image_dict_list[view_i] = view_image_dict_list_view_i
                        all_stable_view_coarse_sampling[view_i] = sampled_views_params

                        
            print("obj_i: ", obj_i, "view_image_dict_list: ", len(view_image_dict_list))

            obj_i_view_image_dict_list = None
            recon_candidates = []
            if not should_add:
                self.move_foundation_model_to_cpu()
                mesh_extracted_list = self.foreground_object_reconstruction(
                    obj_i, view_image_dict_list, total_iterations=500,
                    # main_loss_weight=5.0
                )
                self.move_foundation_model_to_gpu()
                obj_i_view_image_dict_list = view_image_dict_list

                stable_recon_here = False

                for mesh_extracted_info in mesh_extracted_list:
                    mesh_extracted, base_threshold, base_offset, base_tolarant_trans = mesh_extracted_info

                    if mesh_extracted is None:
                        print("mesh_extracted is None")
                        continue

                    all_mesh_list[obj_i] = mesh_extracted
                    try:
                        all_mesh_list = self.instance_meshes_post_pruning_selected(self.start_epoch,
                                                                            input_meshes=all_mesh_list,
                                                                            selected_idx=obj_i)
                    except:
                        print("instance_meshes_post_pruning_selected failed")
                    mesh_extracted = all_mesh_list[obj_i]

                    # debug
                    trimesh.exchange.export.export_mesh(mesh_extracted, os.path.join(self.plots_dir, f"{parent_idx:0>2d}_recon_1.ply"))

                    relative_orientation, relative_trans = sim_validation(coarse_mesh_list_for_phy_test + [mesh_extracted], return_trans=True)
                    print("finish reconstruction for obj_i: ", obj_i, "relative_orientation: ", relative_orientation)
                    print("base_threshold: ", base_threshold, "base_tolarant_trans: ", base_tolarant_trans)
                    print("relative_trans: ", relative_trans, "relative to base offset: ", np.linalg.norm(relative_trans) / base_offset)

                    if relative_orientation >= stable_orientation_threshold:
                        all_mesh_list[obj_i] = mesh_extracted = clean_mesh_floaters_adjust(mesh_extracted)

                        # debug
                        trimesh.exchange.export.export_mesh(mesh_extracted, os.path.join(self.plots_dir, f"{parent_idx:0>2d}_recon_2.ply"))

                        relative_orientation, relative_trans = sim_validation(
                            coarse_mesh_list_for_phy_test + [mesh_extracted], return_trans=True)
                        print("finish reconstruction for obj_i: ", obj_i, "relative_orientation: ",
                              relative_orientation)
                        print("base_threshold: ", base_threshold, "base_tolarant_trans: ", base_tolarant_trans)
                        print("relative_trans: ", relative_trans, "relative to base offset: ",
                              np.linalg.norm(relative_trans) / base_offset)

                    if relative_orientation < stable_orientation_threshold:
                        stable_recon_here = True
                        if base_tolarant_trans > np.linalg.norm(relative_trans) / base_offset:
                            self.mesh_coarse_recon_dict[obj_i] = all_mesh_list[obj_i] = mesh_extracted
                            break
                    else:
                        print("not stable reconstruction for obj_i: ", obj_i, "adding to recon_candidates")
                        recon_candidates.append({
                            "relative_orientation": relative_orientation,
                            "mesh": mesh_extracted,
                        })

                if not stable_recon_here:
                    print("failed to get a stable reconstruction for obj_i: ", obj_i)
                    failed_object_list.append(obj_i)

            else:
                # wonder3d used
                stable_recon = False
                view_choose_opt = [list(range(num_pick_views))] + list(range(num_pick_views))
                print("view_choose_opt: ", view_choose_opt)
                for opt_i, opt in enumerate(view_choose_opt):
                    if opt_i == 0:
                        supp_view_list = []
                        for opt_view_i in opt:
                            if opt_view_i in all_stable_view_image_dict_list:
                                supp_view_list = supp_view_list + all_stable_view_image_dict_list[opt_view_i]


                        self.move_foundation_model_to_cpu()
                        mesh_extracted_list = self.foreground_object_reconstruction(obj_i, view_image_dict_list + supp_view_list, total_iterations=500)
                        self.move_foundation_model_to_gpu()
                        obj_i_view_image_dict_list = view_image_dict_list + supp_view_list

                        print("mesh_extracted_list: ", len(mesh_extracted_list))

                        stable_recon_here = False

                        for mesh_extracted_info in mesh_extracted_list:
                            mesh_extracted, base_threshold, base_offset, base_tolarant_trans = mesh_extracted_info
                            print("mesh_extracted: ", mesh_extracted, "base_threshold: ", base_threshold, "base_offset: ", base_offset, "base_tolarant_trans: ", base_tolarant_trans)

                            if mesh_extracted is None:
                                print("mesh_extracted is None")
                                continue

                            all_mesh_list[obj_i] = mesh_extracted
                            try:
                                all_mesh_list = self.instance_meshes_post_pruning_selected(self.start_epoch,
                                                                                    input_meshes=all_mesh_list,
                                                                                    selected_idx=obj_i)
                            except:
                                print("instance_meshes_post_pruning_selected failed")
                            mesh_extracted = all_mesh_list[obj_i]
                            
                            # debug
                            trimesh.exchange.export.export_mesh(mesh_extracted, os.path.join(self.plots_dir,
                                                                                             f"{parent_idx:0>2d}_recon_1.ply"))

                            relative_orientation, relative_trans = sim_validation(
                                coarse_mesh_list_for_phy_test + [mesh_extracted], return_trans=True)
                            print("finish reconstruction for obj_i: ", obj_i, "relative_orientation: ",
                                  relative_orientation)
                            print("base_threshold: ", base_threshold, "base_tolarant_trans: ", base_tolarant_trans)
                            print("relative_trans: ", relative_trans, "relative to base offset: ",
                                  np.linalg.norm(relative_trans) / base_offset)

                            if relative_orientation >= stable_orientation_threshold:
                                all_mesh_list[obj_i] = mesh_extracted = clean_mesh_floaters_adjust(mesh_extracted)

                                # debug
                                trimesh.exchange.export.export_mesh(mesh_extracted, os.path.join(self.plots_dir,
                                                                                                 f"{parent_idx:0>2d}_recon_2.ply"))

                                relative_orientation, relative_trans = sim_validation(
                                    coarse_mesh_list_for_phy_test + [mesh_extracted], return_trans=True)
                                print("finish reconstruction for obj_i: ", obj_i, "relative_orientation: ",
                                      relative_orientation)
                                print("base_threshold: ", base_threshold, "base_tolarant_trans: ", base_tolarant_trans)
                                print("relative_trans: ", relative_trans, "relative to base offset: ",
                                      np.linalg.norm(relative_trans) / base_offset)

                            if relative_orientation < stable_orientation_threshold:
                                stable_recon_here = True
                                if base_tolarant_trans > np.linalg.norm(relative_trans) / base_offset:
                                    self.mesh_coarse_recon_dict[obj_i] = all_mesh_list[obj_i] = mesh_extracted
                                    break
                            else:
                                print("not stable reconstruction for obj_i: ", obj_i, "adding to recon_candidates")
                                recon_candidates.append({
                                    "relative_orientation": relative_orientation,
                                    "mesh": mesh_extracted,
                                })

                        if stable_recon_here:
                            stable_recon = True
                            break
                        else:
                            print("failed to get a stable reconstruction for obj_i: ", obj_i)
                    else:

                        if opt in all_stable_view_coarse_sampling:

                            print("opt: ", opt)


                            sampled_views = self.sampling_views_around_coarse_recon(all_stable_view_coarse_sampling[opt][0],
                                                                                    all_stable_view_coarse_sampling[opt][1],
                                                                                    all_stable_view_coarse_sampling[opt][2],
                                                                                    all_stable_view_coarse_sampling[opt][3],
                                                                                    all_stable_view_coarse_sampling[opt][4],
                                                                                    all_stable_view_coarse_sampling[opt][5],
                                                                                    all_stable_view_coarse_sampling[opt][6],
                                                                                    all_stable_view_coarse_sampling[opt][7])

                            self.move_foundation_model_to_cpu()
                            mesh_extracted_list = self.foreground_object_reconstruction(obj_i, sampled_views,
                                                     total_iterations=2500, main_loss_weight=20.0,
                                                     start_collision_loss_ratio=0.7)
                            self.move_foundation_model_to_gpu()
                            obj_i_view_image_dict_list = sampled_views

                            stable_recon_here = False

                            print("mesh_extracted_list: ", len(mesh_extracted_list))

                            for mesh_extracted_info in mesh_extracted_list:
                                mesh_extracted, base_threshold, base_offset, base_tolarant_trans = mesh_extracted_info

                                if mesh_extracted is None:
                                    print("mesh_extracted is None")
                                    continue

                                all_mesh_list[obj_i] = mesh_extracted
                                try:
                                    all_mesh_list = self.instance_meshes_post_pruning_selected(self.start_epoch,
                                                                                        input_meshes=all_mesh_list,
                                                                                        selected_idx=obj_i)
                                except:
                                    print("instance_meshes_post_pruning_selected failed")
                                mesh_extracted = all_mesh_list[obj_i]


                                # debug
                                trimesh.exchange.export.export_mesh(mesh_extracted, os.path.join(self.plots_dir,
                                                                                                 f"{parent_idx:0>2d}_recon_1.ply"))

                                relative_orientation, relative_trans = sim_validation(
                                    coarse_mesh_list_for_phy_test + [mesh_extracted], return_trans=True)
                                print("finish reconstruction for obj_i: ", obj_i, "relative_orientation: ",
                                      relative_orientation)
                                print("base_threshold: ", base_threshold, "base_tolarant_trans: ", base_tolarant_trans)
                                print("relative_trans: ", relative_trans, "relative to base offset: ",
                                      np.linalg.norm(relative_trans) / base_offset)

                                if relative_orientation >= stable_orientation_threshold:
                                    all_mesh_list[obj_i] = mesh_extracted = clean_mesh_floaters_adjust(mesh_extracted)

                                    # debug
                                    trimesh.exchange.export.export_mesh(mesh_extracted, os.path.join(self.plots_dir,
                                                                                                     f"{parent_idx:0>2d}_recon_2.ply"))

                                    relative_orientation, relative_trans = sim_validation(
                                        coarse_mesh_list_for_phy_test + [mesh_extracted], return_trans=True)
                                    print("finish reconstruction for obj_i: ", obj_i, "relative_orientation: ",
                                          relative_orientation)
                                    print("base_threshold: ", base_threshold, "base_tolarant_trans: ",
                                          base_tolarant_trans)
                                    print("relative_trans: ", relative_trans, "relative to base offset: ",
                                          np.linalg.norm(relative_trans) / base_offset)

                                if relative_orientation < stable_orientation_threshold:
                                    stable_recon_here = True
                                    if base_tolarant_trans > np.linalg.norm(relative_trans) / base_offset:
                                        self.mesh_coarse_recon_dict[obj_i] = all_mesh_list[obj_i] = mesh_extracted
                                        break
                                else:
                                    print("not stable reconstruction for obj_i: ", obj_i, "adding to recon_candidates")
                                    recon_candidates.append({
                                        "relative_orientation": relative_orientation,
                                        "mesh": mesh_extracted,
                                    })

                            if stable_recon_here:
                                stable_recon = True
                                break
                            else:
                                print("failed to get a stable reconstruction for obj_i: ", obj_i)

                if not stable_recon:
                    # assert False, "failed to get a stable reconstruction for obj_i: " + str(obj_i)
                    print("failed to get a stable reconstruction for obj_i: ", obj_i)
                    failed_object_list.append(obj_i)

            print("exporting mesh for obj_i: ", obj_i)
            
            if obj_i in failed_object_list:
                # choose the mesh in recon_candidates with the lowest relative orientation, select that item from the list
                recon_candidates = sorted(recon_candidates, key=lambda x: x["relative_orientation"])
                print("number of recon_candidates: ", len(recon_candidates))
                if len(recon_candidates) > 0:
                    self.mesh_coarse_recon_dict[obj_i] = recon_candidates[0]["mesh"]
                    all_mesh_list[obj_i] = self.mesh_coarse_recon_dict[obj_i]

            trimesh.exchange.export.export_mesh(
                self.mesh_coarse_recon_dict[obj_i],
                os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}.ply")
            )

            assert obj_i_view_image_dict_list is not None
            obj_i_view_image_dict_list_save_path = os.path.join(self.plots_dir, f"vis_info_{obj_i}.pkl")
            with open(obj_i_view_image_dict_list_save_path, "wb") as f:
                pickle.dump(obj_i_view_image_dict_list, f)

            all_mesh_dict[obj_i] = self.mesh_coarse_recon_dict[obj_i]

            torch.cuda.empty_cache()

            del view_image_dict_list
            del all_stable_view_image_dict_list
            del all_stable_view_coarse_sampling
            gc.collect()

        print("failed_object_list: ", failed_object_list, "len: ", len(failed_object_list))

        sim_mesh_list = self.solve_intersection()
        print("sim_mesh_list: ", len(sim_mesh_list))
        sim_scene(sim_mesh_list)

    def get_all_meshes(self, epoch, lcc=False):
        obj_mesh_dict = {}
        num_objs = self.model.implicit_network.d_out
        for obj_i in range(num_objs):
            obj_i_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_{obj_i}.ply')
            if os.path.exists(obj_i_mesh_path):
                obj_i_mesh = trimesh.exchange.load.load_mesh(obj_i_mesh_path)

                # get lcc
                if lcc:
                    obj_i_mesh_verts = obj_i_mesh.vertices
                    obj_i_mesh_faces = obj_i_mesh.faces
                    obj_i_mesh_verts, obj_i_mesh_faces, verts_map, faces_map = get_lcc_mesh(obj_i_mesh_verts, obj_i_mesh_faces)
                    obj_i_mesh = trimesh.Trimesh(vertices=obj_i_mesh_verts, faces=obj_i_mesh_faces, process=False)

                obj_mesh_dict[obj_i] = obj_i_mesh
        return obj_mesh_dict

    def get_view_weights_of_subset_meshes_with_training_views_backface_discount_limited_phi(self, obj_mesh_dict, subset_idxs):
        faces_cnts = [0]
        all_objs_meshes = []

        num_objs = self.model.implicit_network.d_out

        subset_mesh = trimesh.util.concatenate([obj_mesh_dict[subidx] for subidx in subset_idxs])
        faces_cnts.append(faces_cnts[-1] + len(subset_mesh.faces))
        all_objs_meshes.append(subset_mesh)

        obj_i_scale, obj_i_center = get_scale_shift(subset_mesh.vertices)
        obj_i_upper_center = obj_i_center + np.array([0, 0, 0.75], dtype=np.float32) * obj_i_scale
        obj_i_lower_center = obj_i_center - np.array([0, 0, 0.75], dtype=np.float32) * obj_i_scale

        for obj_i in range(num_objs):
            if obj_i in subset_idxs:
                continue
            if obj_i not in obj_mesh_dict:
                continue

            obj_i_mesh = obj_mesh_dict[obj_i]
            all_objs_meshes.append(obj_i_mesh)
            faces_cnts.append(faces_cnts[-1] + len(obj_i_mesh.faces))

        all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)
        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
        }

        face_normals = get_faces_normal(all_objs_mesh.vertices, all_objs_mesh.faces)
        face_normals = torch.from_numpy(face_normals).cuda().float()

        # mvps = self.train_dataset.mvps
        poses = torch.stack(self.train_dataset.pose_all, dim=0)

        num_azimuth_bins = 36
        num_altitude_bins = 15

        thetas_base = np.linspace(0., 360., num=num_azimuth_bins)
        phis_base = np.linspace(60., 120., num=num_altitude_bins)
        # theta_select = np.zeros((num_azimuth_bins), dtype=np.bool_)
        #
        # subset_idxs_t = torch.tensor(subset_idxs).int()
        # for camera_i in range(poses.shape[0]):
        #     pose = poses[camera_i]
        #     tvec = pose[:3, 3].reshape(3).cpu().numpy()
        #     instance_mask_i = self.train_dataset.semantic_images[camera_i].int().reshape(-1)
        #     if torch.any(torch.isin(instance_mask_i, subset_idxs_t)):
        #         theta_view, phi_view = get_theta_phi(tvec, obj_i_center.reshape(3))
        #         delta_thetas = np.linspace(-45., 45., num=num_azimuth_bins // 4 + 1)
        #         for delta_theta in delta_thetas:
        #             abs_theta = (theta_view + delta_theta) % 360
        #             theta_select[np.abs(thetas_base - abs_theta) <= 360 / num_azimuth_bins] = True

        angle_view_select = np.zeros((num_azimuth_bins, num_altitude_bins), dtype=np.bool_)
        subset_idxs_t = torch.tensor(subset_idxs).int()
        for camera_i in range(poses.shape[0]):
            pose = poses[camera_i]
            tvec = pose[:3, 3].reshape(3).cpu().numpy()
            instance_mask_i = self.train_dataset.semantic_images[camera_i].int().reshape(-1)
            if torch.any(torch.isin(instance_mask_i, subset_idxs_t)):
                theta_view, phi_view = get_theta_phi(tvec, obj_i_center.reshape(3))
                delta_phis = np.linspace(-15., 15., num=num_altitude_bins // 2 + 1)
                delta_thetas = np.linspace(-45., 45., num=num_azimuth_bins // 4 + 1)
                for delta_theta in delta_thetas:
                    for delta_phi in delta_phis:
                        abs_theta = (theta_view + delta_theta) % 360
                        abs_phi = max(min(phi_view + delta_phi, 120), 60)
                        theta_select = np.abs(thetas_base - abs_theta) <= 360 / num_azimuth_bins
                        phi_select = np.abs(phis_base - abs_phi) <= 60 / num_altitude_bins
                        angle_view_select[theta_select, phi_select] = True

        thetas = []
        phis = []

        mvps = []
        poses = []

        # for theta_view in thetas_base[theta_select].tolist():
        #     for phi_view in phis_base:
        angle_view_select_coords = np.nonzero(angle_view_select)
        for theta_view_i, phi_view_i in zip(angle_view_select_coords[0], angle_view_select_coords[1]):
            theta_view = thetas_base[theta_view_i]
            phi_view = phis_base[phi_view_i]
            radius = 1.0

            camera_x = radius * np.sin(phi_view * np.pi / 180) * np.cos(theta_view * np.pi / 180)
            camera_y = radius * np.sin(phi_view * np.pi / 180) * np.sin(theta_view * np.pi / 180)
            camera_z = radius * np.cos(phi_view * np.pi / 180)
            camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
            camera_pos = apply_inv_scale_shift(camera_pos, obj_i_scale, obj_i_center)

            up = np.array([0, 0, 1], dtype=np.float32)
            lookat = np.array([0, 0, 0], dtype=np.float32)
            lookat = apply_inv_scale_shift(lookat, obj_i_scale, obj_i_center)

            pose = build_camera_matrix(
                torch.from_numpy(camera_pos).float(),
                torch.from_numpy(lookat).float(),
                torch.from_numpy(up).float()
            )

            cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=obj_i_scale)
            cam_proj = torch.from_numpy(cam_proj).float()

            mvp = cam_proj @ torch.inverse(pose.clone())

            mvps.append(mvp)
            poses.append(pose)
            thetas.append(theta_view)
            phis.append(phi_view)

        poses = torch.stack(poses, dim=0)

        camera_locs = poses[..., :3, 3].reshape(-1, 3)
        num_cameras = len(mvps)
        resolution = self.img_res

        triangle_vis = torch.zeros(num_cameras, all_objs_mesh.faces.shape[0], dtype=torch.bool)
        triangle_back_vis = torch.zeros(num_cameras, all_objs_mesh.faces.shape[0], dtype=torch.bool)

        for camera_i in range(len(mvps)):
            mvp = mvps[camera_i].to("cuda")
            pose = poses[camera_i].to("cuda")
            valid, triangle_id, depth = rasterize_mesh(mesh_dict, mvp, self.glctx, resolution)
            cam_normals = get_cam_normal_from_rast(valid, triangle_id, face_normals, pose)
            frontface_pixels = cam_normals[..., 2] < 0
            backface_pixels = cam_normals[..., 2] >= 0

            # triangle_vis_frame_i = torch.unique(triangle_id[valid]).reshape(-1)
            triangle_vis_frame_i = torch.unique(triangle_id[torch.logical_and(valid, frontface_pixels)]).reshape(-1)
            triangle_vis[camera_i, triangle_vis_frame_i] = True

            triangle_back_vis_frame_i = torch.unique(triangle_id[torch.logical_and(valid, backface_pixels)]).reshape(-1)
            triangle_back_vis[camera_i, triangle_back_vis_frame_i] = True


        azimuth_bins = torch.from_numpy(np.linspace(0., 360., num=num_azimuth_bins + 1))
        altitude_bins = torch.from_numpy(np.linspace(60., 120., num=num_altitude_bins + 1))

        obj_i = 0
        obj_i_center = torch.from_numpy(obj_i_center).reshape(1, 3)
        visible_faces = triangle_vis[:, faces_cnts[obj_i]:faces_cnts[obj_i + 1]]
        visible_back_faces = triangle_back_vis[:, faces_cnts[obj_i]:faces_cnts[obj_i + 1]]

        # get thetas and phis
        thetas = torch.tensor(thetas)
        phis = torch.tensor(phis)

        # get thetas and phis bins
        thetas_bins = torch.clip(torch.bucketize(thetas, azimuth_bins) - 1, 0, num_azimuth_bins - 1)
        phis_bins = torch.clip(torch.bucketize(phis, altitude_bins) - 1, 0, num_altitude_bins - 1)

        view_weights = torch.zeros(num_azimuth_bins, num_altitude_bins)
        faces_seen_by_partition = [set() for _ in range(num_azimuth_bins * num_altitude_bins)]
        back_faces_seen_by_partition = [set() for _ in range(num_azimuth_bins * num_altitude_bins)]
        for cam_i in range(num_cameras):
            theta_idx = thetas_bins[cam_i]
            phi_idx = phis_bins[cam_i]
            partition_idx = theta_idx * num_altitude_bins + phi_idx

            visible_faces_cam_i = visible_faces[cam_i]  # f_i,j
            faces_seen_by_partition[partition_idx].update(torch.nonzero(visible_faces_cam_i).reshape(-1).tolist())

            visible_faces_back_cam_i = visible_back_faces[cam_i]  # f_i,j
            back_faces_seen_by_partition[partition_idx].update(torch.nonzero(visible_faces_back_cam_i).reshape(-1).tolist())

        for partition_idx in range(num_azimuth_bins * num_altitude_bins):
            faces_seen = faces_seen_by_partition[partition_idx]
            back_faces_seen = back_faces_seen_by_partition[partition_idx]
            view_weights[partition_idx // num_altitude_bins, partition_idx % num_altitude_bins] = max(len(faces_seen) - len(back_faces_seen) * 10, 0)

        # view_weights = view_weights.reshape(num_azimuth_bins, 1, num_altitude_bins).expand(-1, 3, -1).reshape(
        #     num_azimuth_bins*3, num_altitude_bins)
        # azimuth_bins = torch.from_numpy(np.linspace(0., 360., num=num_azimuth_bins*3 + 1))

        view_weights = view_weights / view_weights.sum()
        view_weights = view_weights.numpy()

        return {
            "azimuth": (azimuth_bins[:-1] + azimuth_bins[1:]) / 2,
            "altitude": (altitude_bins[:-1] + altitude_bins[1:]) / 2,
            "view_weights": view_weights,
            "scale": float(obj_i_scale),
            "center": obj_i_center.cpu().numpy()
        }

    def get_view_weights_of_subset_meshes_with_training_views_backface_discount_limited_phi_strict_bbox(self, obj_mesh_dict, subset_idxs):
        faces_cnts = [0]
        all_objs_meshes = []

        num_objs = self.model.implicit_network.d_out

        subset_mesh = trimesh.util.concatenate([obj_mesh_dict[subidx] for subidx in subset_idxs])
        faces_cnts.append(faces_cnts[-1] + len(subset_mesh.faces))
        all_objs_meshes.append(subset_mesh)

        obj_i_scale, obj_i_center = get_scale_shift(subset_mesh.vertices)
        obj_i_upper_center = obj_i_center + np.array([0, 0, 0.75], dtype=np.float32) * obj_i_scale
        obj_i_lower_center = obj_i_center - np.array([0, 0, 0.75], dtype=np.float32) * obj_i_scale

        for obj_i in range(num_objs):
            if obj_i in subset_idxs:
                continue
            if obj_i not in obj_mesh_dict:
                continue

            obj_i_mesh = obj_mesh_dict[obj_i]
            all_objs_meshes.append(obj_i_mesh)
            faces_cnts.append(faces_cnts[-1] + len(obj_i_mesh.faces))

        all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)
        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
        }

        face_normals = get_faces_normal(all_objs_mesh.vertices, all_objs_mesh.faces)
        face_normals = torch.from_numpy(face_normals).cuda().float()

        # mvps = self.train_dataset.mvps
        poses = torch.stack(self.train_dataset.pose_all, dim=0)

        num_azimuth_bins = 36
        num_altitude_bins = 15

        thetas_base = np.linspace(0., 360., num=num_azimuth_bins)
        phis_base = np.linspace(60., 120., num=num_altitude_bins)
        # theta_select = np.zeros((num_azimuth_bins), dtype=np.bool_)
        #
        # subset_idxs_t = torch.tensor(subset_idxs).int()
        # for camera_i in range(poses.shape[0]):
        #     pose = poses[camera_i]
        #     tvec = pose[:3, 3].reshape(3).cpu().numpy()
        #     instance_mask_i = self.train_dataset.semantic_images[camera_i].int().reshape(-1)
        #     if torch.any(torch.isin(instance_mask_i, subset_idxs_t)):
        #         theta_view, phi_view = get_theta_phi(tvec, obj_i_center.reshape(3))
        #         delta_thetas = np.linspace(-45., 45., num=num_azimuth_bins // 4 + 1)
        #         for delta_theta in delta_thetas:
        #             abs_theta = (theta_view + delta_theta) % 360
        #             theta_select[np.abs(thetas_base - abs_theta) <= 360 / num_azimuth_bins] = True

        angle_view_select = np.zeros((num_azimuth_bins, num_altitude_bins), dtype=np.bool_)
        subset_idxs_t = torch.tensor(subset_idxs).int()
        for camera_i in range(poses.shape[0]):
            pose = poses[camera_i]
            tvec = pose[:3, 3].reshape(3).cpu().numpy()
            instance_mask_i = self.train_dataset.semantic_images[camera_i].int().reshape(-1)
            if torch.any(torch.isin(instance_mask_i, subset_idxs_t)):
                if tvec.reshape(3)[2] > obj_i_upper_center.reshape(3)[2]:
                    theta_view, phi_view = get_theta_phi(tvec, obj_i_upper_center.reshape(3))
                    delta_phis = np.linspace(-8., 8., num=num_altitude_bins // 2 + 1)
                elif tvec.reshape(3)[2] < obj_i_lower_center.reshape(3)[2]:
                    theta_view, phi_view = get_theta_phi(tvec, obj_i_lower_center.reshape(3))
                    delta_phis = np.linspace(-8., 8., num=num_altitude_bins // 2 + 1)
                else:
                    theta_view, _ = get_theta_phi(tvec, obj_i_center.reshape(3))
                    phi_view = 90.
                    delta_phis = [0.]
                delta_thetas = np.linspace(-45., 45., num=num_azimuth_bins // 4 + 1)
                for delta_theta in delta_thetas:
                    for delta_phi in delta_phis:
                        abs_theta = (theta_view + delta_theta) % 360
                        abs_phi = max(min(phi_view + delta_phi, 120), 60)
                        theta_select = np.abs(thetas_base - abs_theta) <= 360 / num_azimuth_bins
                        phi_select = np.abs(phis_base - abs_phi) <= 60 / num_altitude_bins
                        angle_view_select[theta_select, phi_select] = True

        thetas = []
        phis = []

        mvps = []
        poses = []

        # for theta_view in thetas_base[theta_select].tolist():
        #     for phi_view in phis_base:
        angle_view_select_coords = np.nonzero(angle_view_select)
        for theta_view_i, phi_view_i in zip(angle_view_select_coords[0], angle_view_select_coords[1]):
            theta_view = thetas_base[theta_view_i]
            phi_view = phis_base[phi_view_i]
            radius = 1.0

            camera_x = radius * np.sin(phi_view * np.pi / 180) * np.cos(theta_view * np.pi / 180)
            camera_y = radius * np.sin(phi_view * np.pi / 180) * np.sin(theta_view * np.pi / 180)
            camera_z = radius * np.cos(phi_view * np.pi / 180)
            camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
            camera_pos = apply_inv_scale_shift(camera_pos, obj_i_scale, obj_i_center)

            up = np.array([0, 0, 1], dtype=np.float32)
            lookat = np.array([0, 0, 0], dtype=np.float32)
            lookat = apply_inv_scale_shift(lookat, obj_i_scale, obj_i_center)

            pose = build_camera_matrix(
                torch.from_numpy(camera_pos).float(),
                torch.from_numpy(lookat).float(),
                torch.from_numpy(up).float()
            )

            cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=obj_i_scale)
            cam_proj = torch.from_numpy(cam_proj).float()

            mvp = cam_proj @ torch.inverse(pose.clone())

            mvps.append(mvp)
            poses.append(pose)
            thetas.append(theta_view)
            phis.append(phi_view)

        poses = torch.stack(poses, dim=0)

        camera_locs = poses[..., :3, 3].reshape(-1, 3)
        num_cameras = len(mvps)
        resolution = self.img_res

        triangle_vis = torch.zeros(num_cameras, all_objs_mesh.faces.shape[0], dtype=torch.bool)
        triangle_back_vis = torch.zeros(num_cameras, all_objs_mesh.faces.shape[0], dtype=torch.bool)

        for camera_i in range(len(mvps)):
            mvp = mvps[camera_i].to("cuda")
            pose = poses[camera_i].to("cuda")
            valid, triangle_id, depth = rasterize_mesh(mesh_dict, mvp, self.glctx, resolution)
            cam_normals = get_cam_normal_from_rast(valid, triangle_id, face_normals, pose)
            frontface_pixels = cam_normals[..., 2] < 0
            backface_pixels = cam_normals[..., 2] >= 0

            # triangle_vis_frame_i = torch.unique(triangle_id[valid]).reshape(-1)
            triangle_vis_frame_i = torch.unique(triangle_id[torch.logical_and(valid, frontface_pixels)]).reshape(-1)
            triangle_vis[camera_i, triangle_vis_frame_i] = True

            triangle_back_vis_frame_i = torch.unique(triangle_id[torch.logical_and(valid, backface_pixels)]).reshape(-1)
            triangle_back_vis[camera_i, triangle_back_vis_frame_i] = True


        azimuth_bins = torch.from_numpy(np.linspace(0., 360., num=num_azimuth_bins + 1))
        altitude_bins = torch.from_numpy(np.linspace(60., 120., num=num_altitude_bins + 1))

        obj_i = 0
        obj_i_center = torch.from_numpy(obj_i_center).reshape(1, 3)
        visible_faces = triangle_vis[:, faces_cnts[obj_i]:faces_cnts[obj_i + 1]]
        visible_back_faces = triangle_back_vis[:, faces_cnts[obj_i]:faces_cnts[obj_i + 1]]

        # get thetas and phis
        thetas = torch.tensor(thetas)
        phis = torch.tensor(phis)

        # get thetas and phis bins
        thetas_bins = torch.clip(torch.bucketize(thetas, azimuth_bins) - 1, 0, num_azimuth_bins - 1)
        phis_bins = torch.clip(torch.bucketize(phis, altitude_bins) - 1, 0, num_altitude_bins - 1)

        view_weights = torch.zeros(num_azimuth_bins, num_altitude_bins)
        faces_seen_by_partition = [set() for _ in range(num_azimuth_bins * num_altitude_bins)]
        back_faces_seen_by_partition = [set() for _ in range(num_azimuth_bins * num_altitude_bins)]
        for cam_i in range(num_cameras):
            theta_idx = thetas_bins[cam_i]
            phi_idx = phis_bins[cam_i]
            partition_idx = theta_idx * num_altitude_bins + phi_idx

            visible_faces_cam_i = visible_faces[cam_i]  # f_i,j
            faces_seen_by_partition[partition_idx].update(torch.nonzero(visible_faces_cam_i).reshape(-1).tolist())

            visible_faces_back_cam_i = visible_back_faces[cam_i]  # f_i,j
            back_faces_seen_by_partition[partition_idx].update(torch.nonzero(visible_faces_back_cam_i).reshape(-1).tolist())

        for partition_idx in range(num_azimuth_bins * num_altitude_bins):
            faces_seen = faces_seen_by_partition[partition_idx]
            back_faces_seen = back_faces_seen_by_partition[partition_idx]
            view_weights[partition_idx // num_altitude_bins, partition_idx % num_altitude_bins] = max(len(faces_seen) - len(back_faces_seen) * 10, 0)

        # view_weights = view_weights.reshape(num_azimuth_bins, 1, num_altitude_bins).expand(-1, 3, -1).reshape(
        #     num_azimuth_bins*3, num_altitude_bins)
        # azimuth_bins = torch.from_numpy(np.linspace(0., 360., num=num_azimuth_bins*3 + 1))

        view_weights = view_weights / view_weights.sum()
        view_weights = view_weights.numpy()

        return {
            "azimuth": (azimuth_bins[:-1] + azimuth_bins[1:]) / 2,
            "altitude": (altitude_bins[:-1] + altitude_bins[1:]) / 2,
            "view_weights": view_weights,
            "scale": float(obj_i_scale),
            "center": obj_i_center.cpu().numpy()
        }



    def get_all_meshes_view_pruned(self, all_mesh_dict):
        faces_cnts = [0]
        all_objs_meshes = []
        all_mesh_view_pruned_dict = {}
        num_objs = self.model.implicit_network.d_out

        for obj_i in range(num_objs):
            assert obj_i in all_mesh_dict

            obj_i_mesh = all_mesh_dict[obj_i]
            faces_cnts.append(faces_cnts[-1] + len(obj_i_mesh.faces))
            all_objs_meshes.append(obj_i_mesh)

        all_objs_mesh = trimesh.util.concatenate(all_objs_meshes)

        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
        }

        mvps = self.train_dataset.mvps
        poses = torch.stack(self.train_dataset.pose_all, dim=0)
        intrinsics = self.train_dataset.intrinsics_all

        num_cameras = len(mvps)
        resolution = self.img_res

        triangle_vis = torch.zeros(all_objs_mesh.faces.shape[0], dtype=torch.bool)
        for camera_i in range(len(mvps)):
            mvp = mvps[camera_i].to("cuda")

            valid, triangle_id, depth = rasterize_mesh(mesh_dict, mvp, self.glctx, (resolution[0]*2, resolution[1]*2))

            triangle_vis_frame_i = torch.unique(triangle_id[valid]).reshape(-1)
            triangle_vis[triangle_vis_frame_i] = True

        num_azimuth_bins = 36
        num_altitude_bins = 25
        azimuth_bins = torch.from_numpy(np.linspace(0., 360., num=num_azimuth_bins + 1))
        altitude_bins = torch.from_numpy(np.linspace(60., 120., num=num_altitude_bins + 1))
        view_angles = np.stack(np.meshgrid(azimuth_bins, altitude_bins, indexing='ij'), axis=-1).reshape(-1, 2)
        view_angles = view_angles * np.pi / 180
        radius = 1.0
        thetas = view_angles[:, 0]
        phis = view_angles[:, 1]

        fov = 70.0
        H_ = W_ = 512
        cx = W_ / 2
        cy = H_ / 2
        fx = fy = 0.5 * W_ / np.tan(fov / 2 * np.pi / 180)
        near = 0.001
        far = 10.0

        mvps_added = []
        cam_pos_added = []

        for obj_i in range(1, num_objs):
            visible_faces = triangle_vis[faces_cnts[obj_i]:faces_cnts[obj_i + 1]]
            mesh_obj_i = all_objs_meshes[obj_i]
            mesh_obj_i_verts = torch.from_numpy(mesh_obj_i.vertices).float()
            mesh_obj_i_faces = torch.from_numpy(mesh_obj_i.faces).long()
            mesh_obj_i_visible_centroids = mesh_obj_i_verts[mesh_obj_i_faces[visible_faces].reshape(-1)].reshape(-1, 3, 3).mean(dim=1)

            scale_i, shift_i = get_scale_shift(mesh_obj_i_visible_centroids.cpu().numpy())
            camera_x = radius * np.sin(phis) * np.cos(thetas)
            camera_y = radius * np.sin(phis) * np.sin(thetas)
            camera_z = radius * np.cos(phis)
            camera_pos = np.stack([camera_x, camera_y, camera_z], axis=-1).reshape(-1, 3).astype(np.float32)
            camera_pos = apply_inv_scale_shift(camera_pos, scale_i, shift_i)

            lookat = np.array([0, 0, 0], dtype=np.float32)
            lookat = apply_inv_scale_shift(lookat, scale_i, shift_i)
            up = np.array([0, 0, 1], dtype=np.float32)

            camera_pos_t = torch.from_numpy(camera_pos).float().cuda()
            camera_pos_sdf_values = self.model.implicit_network.get_sdf_vals(camera_pos_t).reshape(-1)
            camera_pos_t = camera_pos_t[camera_pos_sdf_values > 0.05]
            cam_pos_added.append(camera_pos_t.cpu().numpy())

            # camproj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=scale_i)

            camproj = get_camera_perspective_projection_matrix(fx, fy, cx, cy, H_, W_, near=near, far=far)
            camproj = torch.from_numpy(camproj).float()

            for camera_i in range(camera_pos_t.shape[0]):
                pose_i = build_camera_matrix(
                    camera_pos_t[camera_i].float().cpu(),
                    torch.from_numpy(lookat).float(),
                    torch.from_numpy(up).float()
                ).float()
                mvp_i = camproj @ torch.inverse(pose_i.clone())
                mvps_added.append(mvp_i)

        if False:
            cam_pos_added = np.concatenate(cam_pos_added, axis=0)
            # save as ply with open3d
            import open3d as o3d
            cam_pos_added_pcd = o3d.geometry.PointCloud()
            cam_pos_added_pcd.points = o3d.utility.Vector3dVector(cam_pos_added)
            o3d.io.write_point_cloud(os.path.join(self.plots_dir, "cam_pos_added.ply"), cam_pos_added_pcd)


        print("rasterizing for mvp added:")
        for camera_i in tqdm(range(len(mvps_added))):
            mvp = mvps_added[camera_i].to("cuda")

            valid, triangle_id, depth = rasterize_mesh(mesh_dict, mvp, self.glctx, (resolution[0]*2, resolution[1]*2))

            triangle_vis_frame_i = torch.unique(triangle_id[valid]).reshape(-1)
            triangle_vis[triangle_vis_frame_i] = True

        for obj_i in range(num_objs):

            visible_faces = triangle_vis[faces_cnts[obj_i]:faces_cnts[obj_i + 1]]
            all_mesh_view_pruned_dict[obj_i] = all_mesh_dict[obj_i].submesh([visible_faces], append=True)

        return all_mesh_view_pruned_dict

    def get_obj_mesh_view_pruned(self, mesh_obj):

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
            triangle_vis_frame_i = torch.unique(triangle_id[valid]).reshape(-1)
            triangle_vis[triangle_vis_frame_i] = True

        mesh_obj_pruned = mesh_obj.submesh([triangle_vis], append=True)
        return mesh_obj_pruned

    def get_bg_sampling(self, all_mesh_dict):

        vis_dir = os.path.join(self.plots_dir, "bg_vis")
        os.makedirs(vis_dir, exist_ok=True)

        bg_occluded_indices, bg_mesh_pruned = self.get_occluded_faces_for_obj(all_mesh_dict, 0)

        triangles = bg_mesh_pruned.vertices[bg_mesh_pruned.faces.reshape(-1)].reshape(-1, 3, 3)
        edge_01_len = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=-1)
        edge_12_len = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=-1)
        edge_20_len = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=-1)
        average_edge_len = np.mean((edge_01_len + edge_12_len + edge_20_len) / 3)

        occluded_pts = bg_mesh_pruned.triangles_center[bg_occluded_indices]

        cluster_pts, cluster_indices = cluster_points(occluded_pts, "dbscan", eps=5*average_edge_len)

        for cluster_i in range(len(cluster_indices)):
            cluster_indices[cluster_i] = bg_occluded_indices[cluster_indices[cluster_i]]

        bg_center = (bg_mesh_pruned.bounds[0] + bg_mesh_pruned.bounds[1]) / 2
        bg_scale = np.max(bg_mesh_pruned.bounds[1] - bg_mesh_pruned.bounds[0])

        samples_offset_x = np.linspace(-0.5 * bg_scale, 0.5 * bg_scale, num=40)
        samples_offset_y = np.linspace(-0.5 * bg_scale, 0.5 * bg_scale, num=40)
        samples_offset_z = np.linspace(-0.5 * bg_scale, 0.5 * bg_scale, num=40)

        samples_offset_pts = np.stack(
            np.meshgrid(samples_offset_x, samples_offset_y, samples_offset_z, indexing='ij'),
            axis=-1).reshape(-1, 3)

        sample_pts = samples_offset_pts + bg_center.reshape(1, 3)

        sample_sdfs = []
        for _pts in torch.split(torch.from_numpy(sample_pts).float(), 1024):
            sample_sdfs.append(self.model.implicit_network.get_object_sdf_vals(_pts.cuda(), 0).cpu().reshape(-1))
        sample_sdfs = torch.cat(sample_sdfs, dim=0)

        sample_cnt = len(cluster_indices)
        sample_idxs = torch.argsort(sample_sdfs)[-sample_cnt:]
        sample_pts = sample_pts[sample_idxs.cpu().numpy()]
        sample_sdfs = sample_sdfs[sample_idxs]
        assert not torch.any(sample_sdfs < 0.), f"sample_sdfs: {sample_sdfs}"

        fov = 80.0 * np.pi / 180
        H, W = 256, 256
        near = 0.001
        far = 10.0
        all_occluded_face_indices = torch.from_numpy(np.concatenate(
            [np.array(cluster_indices[cluster_i], dtype=np.int32).reshape(-1) for cluster_i in range(len(cluster_indices))],
            axis=0))

        bg_views_dict_list = []

        for cluster_i in range(sample_cnt):
            torch.cuda.empty_cache()
            camera_pos = sample_pts[cluster_i].reshape(3)
            lookat = np.mean(cluster_pts[cluster_i], axis=0).reshape(3)
            up = np.array([0, 0, 1], dtype=np.float32)

            pose = build_camera_matrix(
                torch.from_numpy(camera_pos).float(),
                torch.from_numpy(lookat).float(),
                torch.from_numpy(up).float()
            ).float()

            fx, fy = fov_to_focal_length(fov, (H, W))
            cx, cy = W / 2, H / 2
            cam_proj = get_camera_perspective_projection_matrix(fx, fy, cx, cy, H, W, near, far)
            cam_proj = torch.from_numpy(cam_proj).float()

            mvp = cam_proj @ torch.inverse(pose.clone())
            mvp = mvp.cuda()

            valid, triangle_id, _ = rasterize_trimesh(bg_mesh_pruned, mvp, self.glctx, (H, W))
            valid = valid.cpu()
            triangle_id = triangle_id.cpu()
            occluded_pixels = torch.logical_and(
                valid.reshape(-1),
                torch.isin(triangle_id.reshape(-1), all_occluded_face_indices)
            ).reshape(H, W).cpu().numpy()

            occluded_pixels = binary_dilation(occluded_pixels, iterations=4)

            ray_origins, ray_dirs = get_camera_perspective_rays_world(fx, fy, cx, cy, H, W, pose.numpy())
            ray_origins = torch.from_numpy(ray_origins.reshape(-1, 3))
            ray_dirs = torch.from_numpy(ray_dirs.reshape(-1, 3))

            rgb_list = []
            nm_list = []
            for ray_o, ray_d in zip(torch.split(ray_origins, 1024), torch.split(ray_dirs, 1024)):
                out = self.model.forward_multi_obj_rays_subset_all_sdf(ray_o.cuda(), ray_d.cuda(), pose.cuda(),
                                                                       [0], [0])

                rgb_list.append(out['rgb_values'].detach().cpu())
                nm_list.append(out['normal_map'].detach().cpu())

            rgb_pred = torch.cat(rgb_list, dim=0)
            nm_pred = torch.cat(nm_list, dim=0)

            rgb_pred = rgb_pred.reshape(H, W, 3).detach().cpu().numpy()
            nm_pred = nm_pred.reshape(H, W, 3).detach().cpu().numpy()

            rgb_pred_masked = rgb_pred.copy()
            rgb_pred_masked[occluded_pixels] = rgb_pred_masked[occluded_pixels] * 0.5 + np.array([1., 0., 0.]) * 0.5

            im_label = f"bg_{cluster_i:0>2d}"

            rgb_path = os.path.join(vis_dir, f"{im_label}_rgb.png")
            nm_path = os.path.join(vis_dir, f"{im_label}_nm.png")
            rgb_masked_path = os.path.join(vis_dir, f"{im_label}_rgb_masked.png")

            Image.fromarray(np.clip(rgb_pred * 255, 0, 255).astype(np.uint8)).save(rgb_path)
            Image.fromarray(np.clip((nm_pred * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)).save(nm_path)
            Image.fromarray(np.clip(rgb_pred_masked * 255, 0, 255).astype(np.uint8)).save(rgb_masked_path)

            rgb_to_inpaint = rgb_pred.copy()
            rgb_to_inpaint[occluded_pixels] = 1.0

            rgb_to_inpaint_path = os.path.join(vis_dir, f"{im_label}_rgb_f1_to_inpaint.png")
            Image.fromarray(np.clip(rgb_to_inpaint * 255, 0, 255).astype(np.uint8)).save(rgb_to_inpaint_path)

            rgb_inpainted = inpaint(self.lama_model, self.lama_predict_config, torch.from_numpy(rgb_to_inpaint),
                                    torch.from_numpy(occluded_pixels))
            rgb_inpainted = rgb_inpainted.cpu().numpy()
            rgb_inpainted_path = os.path.join(vis_dir, f"{im_label}_rgb_f2_inpainted.png")
            Image.fromarray(np.clip(rgb_inpainted * 255, 0, 255).astype(np.uint8)).save(
                rgb_inpainted_path)

            normal_inpainted = infer_normal(self.omnidata_normal_model, rgb_inpainted)
            normal_inpainted = align_normal_pred_lama_omnidata(normal_inpainted, nm_pred, np.logical_not(occluded_pixels))

            normal_inpainted_path = os.path.join(vis_dir, f"{im_label}_normal_inpainted.png")
            Image.fromarray(np.clip((normal_inpainted + 1) * 0.5 * 255, 0, 255).astype(np.uint8)).save(
                normal_inpainted_path)

            bg_views_dict_list.append({
                "rgb": rgb_inpainted,
                "normal": normal_inpainted,
                "mask": occluded_pixels,
                "pose": pose.cpu(),
                "intrinsics": [fx, fy, cx, cy]
            })

        return bg_views_dict_list


    def background_inpainting_sampling(self, all_mesh_dict):

        vis_dir = os.path.join(self.plots_dir, "bg_vis")
        os.makedirs(vis_dir, exist_ok=True)

        bg_occluded_indices, bg_mesh_pruned = self.get_occluded_faces_for_obj(all_mesh_dict, 0)

        triangles = bg_mesh_pruned.vertices[bg_mesh_pruned.faces.reshape(-1)].reshape(-1, 3, 3)
        edge_01_len = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=-1)
        edge_12_len = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=-1)
        edge_20_len = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=-1)
        average_edge_len = np.mean((edge_01_len + edge_12_len + edge_20_len) / 3)

        occluded_pts = bg_mesh_pruned.triangles_center[bg_occluded_indices]

        cluster_pts, cluster_indices = cluster_points(occluded_pts, "dbscan", eps=5*average_edge_len)

        for cluster_i in range(len(cluster_indices)):
            cluster_indices[cluster_i] = bg_occluded_indices[cluster_indices[cluster_i]]

        all_occluded_face_indices = torch.from_numpy(np.concatenate(
            [np.array(cluster_indices[cluster_i], dtype=np.int32).reshape(-1) for cluster_i in range(len(cluster_indices))],
            axis=0))

        cluster_seen_dict = {}
        mvps = self.train_dataset.mvps
        poses = self.train_dataset.pose_all
        H, W = self.train_dataset.img_res

        num_images = len(mvps)
        for image_i in range(num_images):
            mvp = mvps[image_i].to("cuda")
            pose = poses[image_i].to("cuda")

            valid, triangle_id, _ = rasterize_trimesh(bg_mesh_pruned, mvp, self.glctx, (H, W))
            valid = valid.cpu()
            triangle_id = triangle_id.cpu()
            occluded_triangle_id = triangle_id[valid].unique().cpu()

            for cluster_i in range(len(cluster_indices)):
                cluster_i_visible = torch.isin(occluded_triangle_id, torch.from_numpy(cluster_indices[cluster_i]))
                if torch.any(cluster_i_visible):
                    if cluster_i not in cluster_seen_dict:
                        cluster_seen_dict[cluster_i] = []
                    cluster_seen_dict[cluster_i].append(image_i)

        # print("cluster_seen_dict: ", cluster_seen_dict)
        sample_views = []

        fov = 80.0 * np.pi / 180
        sample_H, sample_W = 256, 256
        near = 0.001
        far = 10.0

        for cluster_i in range(len(cluster_indices)):
            if cluster_i not in cluster_seen_dict:
                print(f"cluster {cluster_i} not seen")
                continue
            cluster_i_seen_idxs = cluster_seen_dict[cluster_i]
            if len(cluster_i_seen_idxs) > 2:
                # randomly pick 2 from it
                cluster_i_seen_idxs = np.random.choice(cluster_i_seen_idxs, 2, replace=False)

            for image_i in cluster_i_seen_idxs:
                camera_pos = poses[image_i][:3, 3].cpu().numpy()
                lookat = np.mean(cluster_pts[cluster_i], axis=0).reshape(3)
                up = np.array([0, 0, 1], dtype=np.float32)

                pose = build_camera_matrix(
                    torch.from_numpy(camera_pos).float(),
                    torch.from_numpy(lookat).float(),
                    torch.from_numpy(up).float()
                ).float()

                fx, fy = fov_to_focal_length(fov, (sample_H, sample_W))
                cx, cy = sample_W / 2, sample_H / 2
                cam_proj = get_camera_perspective_projection_matrix(fx, fy, cx, cy, sample_H, sample_W, near, far)
                cam_proj = torch.from_numpy(cam_proj).float()

                mvp = cam_proj @ torch.inverse(pose.clone())

                sample_views.append({
                    "pose": pose.cpu(),
                    "intrinsics": [fx, fy, cx, cy],
                    "mvp": mvp
                })

        sample_cnt = len(sample_views)

        bg_views_dict_list = []

        # sampling in bg views
        print("sampling in bg views...")
        for sample_i in tqdm(range(sample_cnt)):
            torch.cuda.empty_cache()

            pose = sample_views[sample_i]["pose"]
            mvp = sample_views[sample_i]["mvp"].cuda()
            fx = sample_views[sample_i]["intrinsics"][0]
            fy = sample_views[sample_i]["intrinsics"][1]
            cx = sample_views[sample_i]["intrinsics"][2]
            cy = sample_views[sample_i]["intrinsics"][3]

            valid, triangle_id, _ = rasterize_trimesh(bg_mesh_pruned, mvp, self.glctx, (sample_H, sample_W))
            valid = valid.cpu()
            triangle_id = triangle_id.cpu()
            occluded_pixels = torch.logical_and(
                valid.reshape(-1),
                torch.isin(triangle_id.reshape(-1), all_occluded_face_indices)
            ).reshape(sample_H, sample_W).cpu().numpy()

            occluded_pixels = binary_dilation(occluded_pixels, iterations=8)

            ray_origins, ray_dirs = get_camera_perspective_rays_world(fx, fy, cx, cy, sample_H, sample_W, pose.numpy())
            ray_origins = torch.from_numpy(ray_origins.reshape(-1, 3))
            ray_dirs = torch.from_numpy(ray_dirs.reshape(-1, 3))

            rgb_list = []
            nm_list = []
            depth_list = []
            for ray_o, ray_d in zip(torch.split(ray_origins, 1024), torch.split(ray_dirs, 1024)):
                out = self.model.forward_multi_obj_rays_subset_all_sdf(ray_o.cuda(), ray_d.cuda(), pose.cuda(),
                                                                       [0], [0])

                rgb_list.append(out['rgb_values'].detach().cpu())
                nm_list.append(out['normal_map'].detach().cpu())
                depth_list.append(out['depth_values'].detach().cpu())

            rgb_pred = torch.cat(rgb_list, dim=0)
            nm_pred = torch.cat(nm_list, dim=0)
            depth_pred = torch.cat(depth_list, dim=0)

            rgb_pred = rgb_pred.reshape(sample_H, sample_W, 3).detach().cpu().numpy()
            nm_pred = nm_pred.reshape(sample_H, sample_W, 3).detach().cpu().numpy()
            depth_pred = depth_pred.reshape(sample_H, sample_W, 1).detach().cpu().numpy()

            rgb_pred_masked = rgb_pred.copy()
            rgb_pred_masked[occluded_pixels] = rgb_pred_masked[occluded_pixels] * 0.5 + np.array([1., 0., 0.]) * 0.5

            im_label = f"bg_{sample_i:0>2d}"

            rgb_path = os.path.join(vis_dir, f"{im_label}_rgb.png")
            nm_path = os.path.join(vis_dir, f"{im_label}_nm.png")
            rgb_masked_path = os.path.join(vis_dir, f"{im_label}_rgb_masked.png")

            Image.fromarray(np.clip(rgb_pred * 255, 0, 255).astype(np.uint8)).save(rgb_path)
            Image.fromarray(np.clip((nm_pred * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)).save(nm_path)
            Image.fromarray(np.clip(rgb_pred_masked * 255, 0, 255).astype(np.uint8)).save(rgb_masked_path)

            rgb_to_inpaint = rgb_pred.copy()
            rgb_to_inpaint[occluded_pixels] = 1.0

            rgb_to_inpaint_path = os.path.join(vis_dir, f"{im_label}_rgb_f1_to_inpaint.png")
            Image.fromarray(np.clip(rgb_to_inpaint * 255, 0, 255).astype(np.uint8)).save(rgb_to_inpaint_path)

            rgb_inpainted = inpaint(self.lama_model, self.lama_predict_config, torch.from_numpy(rgb_to_inpaint),
                                    torch.from_numpy(occluded_pixels))
            rgb_inpainted = rgb_inpainted.cpu().numpy()
            rgb_inpainted_path = os.path.join(vis_dir, f"{im_label}_rgb_f2_inpainted.png")
            Image.fromarray(np.clip(rgb_inpainted * 255, 0, 255).astype(np.uint8)).save(
                rgb_inpainted_path)

            # normal_inpainted = infer_normal(self.omnidata_normal_model, rgb_inpainted)
            # normal_inpainted = align_normal_pred_lama_omnidata(normal_inpainted, nm_pred, np.logical_not(occluded_pixels))
            #
            # normal_inpainted_path = os.path.join(vis_dir, f"{im_label}_normal_inpainted.png")
            # Image.fromarray(np.clip((normal_inpainted + 1) * 0.5 * 255, 0, 255).astype(np.uint8)).save(
            #     normal_inpainted_path)

            nm_to_inpaint = nm_pred.copy()
            nm_to_inpaint = nm_to_inpaint * 0.5 + 0.5
            nm_to_inpaint[occluded_pixels] = 1.0
            nm_inpainted_from_lama_raw = inpaint(self.lama_model, self.lama_predict_config,
                                                 torch.from_numpy(nm_to_inpaint),
                                                 torch.from_numpy(occluded_pixels))
            normal_inpainted = nm_inpainted_from_lama_raw * 2 - 1
            normal_inpainted = normal_inpainted.cpu().numpy()

            normal_inpainted_path = os.path.join(vis_dir, f"{im_label}_normal_inpainted.png")
            Image.fromarray(np.clip((normal_inpainted + 1) * 0.5 * 255, 0, 255).astype(np.uint8)).save(
                normal_inpainted_path)

            depth_to_inpaint = np.concatenate([depth_pred.copy()] * 3, axis=-1)
            fg_region_lama = np.logical_not(occluded_pixels)
            depth_min = np.min(depth_to_inpaint[fg_region_lama]) - 0.1
            depth_max = np.max(depth_to_inpaint[fg_region_lama]) + 0.1
            depth_to_inpaint = (depth_to_inpaint - depth_min) / (depth_max - depth_min)
            depth_to_inpaint[occluded_pixels] = 1.0
            depth_inpainted_from_lama_raw = inpaint(self.lama_model, self.lama_predict_config,
                                                 torch.from_numpy(depth_to_inpaint),
                                                 torch.from_numpy(occluded_pixels))
            depth_inpainted_from_lama = depth_inpainted_from_lama_raw.cpu().numpy()
            depth_inpainted_from_lama = depth_inpainted_from_lama.mean(axis=-1).reshape(sample_H, sample_W)
            depth_inpainted_from_lama_vis = depth_inpainted_from_lama.copy()
            depth_inpainted = depth_inpainted_from_lama * (depth_max - depth_min) + depth_min

            Image.fromarray(np.clip(
                np.concatenate([depth_inpainted_from_lama_vis[..., None]] * 3, axis=-1) * 255, 0, 255).astype(np.uint8)).save(
                os.path.join(vis_dir, f"{im_label}_depth_inpainted.png"))

            bg_views_dict_list.append({
                "rgb": rgb_inpainted,
                "normal": normal_inpainted,
                "depth": depth_inpainted,
                "mask": occluded_pixels,
                "pose": pose.cpu(),
                "intrinsics": [fx, fy, cx, cy]
            })

        return bg_views_dict_list


    def get_occluded_faces_for_obj(self, all_mesh_dict, obj_i):
        num_objs = self.model.implicit_network.d_out

        mesh_obj_i = all_mesh_dict[obj_i]
        mesh_obj_i_pruned = self.get_obj_mesh_view_pruned(mesh_obj_i)

        other_meshes = [all_mesh_dict[obj_j] for obj_j in range(num_objs) if obj_j != obj_i and obj_j in all_mesh_dict]
        other_mesh = trimesh.util.concatenate(other_meshes)

        occluded_face_indices = occlusion_test(mesh_obj_i_pruned, other_mesh)

        return occluded_face_indices, mesh_obj_i_pruned

    def background_inpainting(self):
        all_mesh_dict = self.get_all_meshes(self.start_epoch)
        return self.background_inpainting_sampling(all_mesh_dict)

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

    def instance_meshes_post_pruning(self, epoch, min_visible_verts=0):
        all_meshes = []
        faces_cnts = [0]
        num_objs = self.model.implicit_network.d_out
        print("num_objs: ", num_objs)
        all_verts = []
        all_faces = []
        num_verts = 0
        for obj_i in range(num_objs):

            obj_i_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_{obj_i}.ply')
            assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
            obj_i_mesh = trimesh.exchange.load.load_mesh(obj_i_mesh_path)

            all_meshes.append(obj_i_mesh)
            faces_cnts.append(faces_cnts[-1] + len(obj_i_mesh.faces))

            all_verts.append(obj_i_mesh.vertices)
            all_faces.append(obj_i_mesh.faces + num_verts)
            num_verts += obj_i_mesh.vertices.shape[0]

        # all_objs_mesh = trimesh.util.concatenate(all_meshes)
        all_objs_mesh = trimesh.Trimesh(vertices=np.concatenate(all_verts, axis=0), faces=np.concatenate(all_faces, axis=0), process=False)
        trimesh.exchange.export.export_mesh(all_objs_mesh, os.path.join(self.plots_dir, f'surface_{epoch}_all_objs.ply'))

        print("all_objs_mesh: ", all_objs_mesh.vertices.shape, all_objs_mesh.faces.shape)

        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
        }

        mvps = self.train_dataset.mvps
        H, W = resolution = self.img_res

        H_sr, W_sr = H, W
        while H_sr % 8 != 0 or W_sr % 8 != 0:
            H_sr *= 2
            W_sr *= 2

        triangle_vis = torch.zeros(all_objs_mesh.faces.shape[0], dtype=torch.bool)

        for cam_i in range(len(mvps)):
            mvp = mvps[cam_i].to("cuda")
            valid, triangle_id, _ = rasterize_mesh(mesh_dict, mvp, self.glctx, (H_sr, W_sr))
            if H_sr != H:
                # print("resizing")
                valid = resize_int_tensor(valid, (H, W))
                triangle_id = resize_int_tensor(triangle_id, (H, W))

            instance_mask_i = self.train_dataset.semantic_images[cam_i].int().to("cuda").reshape(H, W)
            valid_objs_mask = torch.zeros((H, W), dtype=torch.bool).to("cuda")
            # instance_mask_i_np = instance_mask_i.cpu().numpy()

            obj_mask_vis = np.ones((H, W, 3))
            obj_sem_mask_vis = np.ones((H, W, 3))

            for obj_i in range(num_objs):
                valid_instance = instance_mask_i == obj_i
                obj_i_mask = torch.logical_and(triangle_id >= faces_cnts[obj_i], triangle_id < faces_cnts[obj_i + 1])
                valid_obj_i = torch.logical_and(valid, valid_instance)
                valid_obj_i = torch.logical_and(valid_obj_i, obj_i_mask)
                valid_objs_mask = torch.logical_or(valid_objs_mask, valid_obj_i)

            triangle_vis_frame_i = torch.unique(triangle_id[valid_objs_mask]).reshape(-1)
            triangle_vis[triangle_vis_frame_i] = True

        for obj_i in range(num_objs):
            visible_faces = triangle_vis[faces_cnts[obj_i]:faces_cnts[obj_i + 1]]

            obj_i_mesh = all_meshes[obj_i]
            verts, faces = obj_i_mesh.vertices, obj_i_mesh.faces

            vertices_seen_indices = np.unique(faces[visible_faces.cpu().numpy()].reshape(-1))

            edges = obj_i_mesh.edges_sorted.reshape((-1, 2))
            components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
            components = [comp for comp in components if len(np.intersect1d(comp, vertices_seen_indices)) > min_visible_verts]
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

            faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
            faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
            faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]

            vert_colors = obj_i_mesh.visual.vertex_colors[..., :3].reshape(-1, 3)
            vert_colors = vert_colors[verts_map]

            all_meshes[obj_i] = trimesh.Trimesh(vertices=verts_lcc, faces=faces_lcc, vertex_colors=vert_colors, process=False)

        return all_meshes

    def instance_meshes_post_pruning_selected(self, epoch, min_visible_verts=0, input_meshes=None, selected_idx=None):
        all_meshes = []
        faces_cnts = [0]
        num_objs = self.model.implicit_network.d_out
        for obj_i in range(num_objs):
            if input_meshes is None:
                obj_i_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_{obj_i}.ply')
                assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
                obj_i_mesh = trimesh.exchange.load.load_mesh(obj_i_mesh_path)
            else:
                obj_i_mesh = input_meshes[obj_i]
            all_meshes.append(obj_i_mesh)
            faces_cnts.append(faces_cnts[-1] + len(obj_i_mesh.faces))

        all_objs_mesh = trimesh.util.concatenate(all_meshes)

        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(all_objs_mesh.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(all_objs_mesh.faces).int().to("cuda").contiguous()
        }

        mvps = self.train_dataset.mvps
        H, W = resolution = self.img_res

        H_sr, W_sr = H, W
        while H_sr % 8 != 0 or W_sr % 8 != 0:
            H_sr *= 2
            W_sr *= 2

        triangle_vis = torch.zeros(all_objs_mesh.faces.shape[0], dtype=torch.bool)

        for cam_i in range(len(mvps)):
            mvp = mvps[cam_i].to("cuda")
            valid, triangle_id, _ = rasterize_mesh(mesh_dict, mvp, self.glctx, (H_sr, W_sr))
            if H_sr != H:
                valid = resize_int_tensor(valid, (H, W))
                triangle_id = resize_int_tensor(triangle_id, (H, W))

            instance_mask_i = self.train_dataset.semantic_images[cam_i].int().to("cuda").reshape(H, W)
            valid_objs_mask = torch.zeros((H, W), dtype=torch.bool).to("cuda")
            for obj_i in range(num_objs):
                valid_instance = instance_mask_i == obj_i
                obj_i_mask = torch.logical_and(triangle_id >= faces_cnts[obj_i], triangle_id < faces_cnts[obj_i + 1])
                valid_obj_i = torch.logical_and(valid, valid_instance)
                valid_obj_i = torch.logical_and(valid_obj_i, obj_i_mask)
                valid_objs_mask = torch.logical_or(valid_objs_mask, valid_obj_i)

            triangle_vis_frame_i = torch.unique(triangle_id[valid_objs_mask]).reshape(-1)
            triangle_vis[triangle_vis_frame_i] = True

        for obj_i in range(num_objs):

            visible_faces = triangle_vis[faces_cnts[obj_i]:faces_cnts[obj_i + 1]]

            obj_i_mesh = all_meshes[obj_i]
            verts, faces = obj_i_mesh.vertices, obj_i_mesh.faces

            # bounding box pruning
            if obj_i > 0:

                original_bbox_obj_i = self.original_bbox_dict[obj_i]
                object_center = original_bbox_obj_i['center']
                object_scale = original_bbox_obj_i['scale']

                out_of_bbox_verts = np.any(np.abs((verts - object_center) / object_scale) > 0.6, axis=-1)
                out_of_bbox_faces_verts = out_of_bbox_verts[faces.reshape(-1)].reshape(-1, 3)

                out_of_bbox_faces = np.any(out_of_bbox_faces_verts, axis=-1)
                out_of_bbox_faces = torch.from_numpy(out_of_bbox_faces)
                visible_faces = torch.logical_and(visible_faces, torch.logical_not(out_of_bbox_faces))

                triangle_vis[faces_cnts[obj_i]:faces_cnts[obj_i + 1]] = visible_faces

        all_objs_mesh_vis_color = np.ones((all_objs_mesh.faces.shape[0], 3))
        all_objs_mesh_vis_color[triangle_vis.cpu().numpy().reshape(-1)] = np.array([1.0, 0.0, 0.0])

        for obj_i in range(num_objs):

            if selected_idx is not None and obj_i != selected_idx:
                continue

            visible_faces = triangle_vis[faces_cnts[obj_i]:faces_cnts[obj_i + 1]]

            obj_i_mesh = all_meshes[obj_i]
            verts, faces = obj_i_mesh.vertices, obj_i_mesh.faces

            vertices_seen_indices = np.unique(faces[visible_faces.cpu().numpy()].reshape(-1))

            edges = obj_i_mesh.edges_sorted.reshape((-1, 2))
            components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
            components = [comp for comp in components if len(np.intersect1d(comp, vertices_seen_indices)) > min_visible_verts]
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

            faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
            faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
            faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]

            vert_colors = obj_i_mesh.visual.vertex_colors[..., :3].reshape(-1, 3)
            vert_colors = vert_colors[verts_map]

            all_meshes[obj_i] = trimesh.Trimesh(vertices=verts_lcc, faces=faces_lcc, vertex_colors=vert_colors, process=False)

        return all_meshes

    def generate_bbox(self, all_meshes):
        bbox_root_path = os.path.join(self.plots_dir, 'bbox')
        os.makedirs(bbox_root_path, exist_ok=True)

        for mesh_i, mesh in enumerate(all_meshes):

            bbox_json_path = os.path.join(bbox_root_path, f'bbox_{mesh_i}.json')
            if os.path.exists(bbox_json_path):
                os.remove(bbox_json_path)
            x_min, x_max = mesh.vertices[:, 0].min() - 0.03, mesh.vertices[:, 0].max() + 0.03
            y_min, y_max = mesh.vertices[:, 1].min() - 0.03, mesh.vertices[:, 1].max() + 0.03
            z_min, z_max = mesh.vertices[:, 2].min() - 0.03, mesh.vertices[:, 2].max() + 0.03
            x_min, x_max = max(x_min, -1.0), min(x_max, 1.0)
            y_min, y_max = max(y_min, -1.0), min(y_max, 1.0)
            z_min, z_max = max(z_min, -1.0), min(z_max, 1.0)
            obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            with open(bbox_json_path, 'w') as f:
                json.dump(obj_bbox, f)
            print(f'bbox_{mesh_i}.json save to {bbox_root_path}')

    def background_reconstruction(self,
                        collision_sample_cnt=64, collision_sample_scale_factor=0.6):
        print("begin bg recon locally")
        conf_model = self.conf.get_config('model')
        local_model = utils.get_class(self.conf.get_string('train.model_class'))(
            conf=conf_model,
            plots_dir=self.plots_dir,
            graph_node_dict=self.train_dataset.graph_node_dict,
            ft_folder=self.ft_folder,
            num_images=len(self.train_dataset.mvps)
        )

        local_optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(local_model.implicit_network.grid_parameters()),
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(local_model.implicit_network.mlp_parameters()) +\
                    list(local_model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(local_model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)

        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        local_scheduler = torch.optim.lr_scheduler.ExponentialLR(local_optimizer, decay_rate ** (1. / decay_steps))

        local_model = local_model.cuda()
        local_model.load_state_dict(copy.deepcopy(self.ckpt_dict['model']))
        local_model.train()

        self.train_dataset.sampling_class_id = 0
        self.train_dataset.change_sampling_idx(self.num_pixels)

        local_optimizer.load_state_dict(copy.deepcopy(self.ckpt_dict['optimizer']))
        local_scheduler.load_state_dict(copy.deepcopy(self.ckpt_dict['scheduler']))
        print("finish loading ckpt to local models")

        total_iterations = 500
        total_epoch = total_iterations // len(self.train_dataloader) + 1

        iter_step = 0
        for epoch in range(total_epoch):
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                # print("into dataloader")
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                local_optimizer.zero_grad()

                model_outputs = local_model(model_input, indices, iter_step=iter_step)
                model_outputs['iter_step'] = iter_step

                loss = torch.tensor(0.0).cuda().float()
                loss_output = {}

                loss_output = self.loss(model_outputs, ground_truth, call_reg=True) if \
                    self.iter_step >= self.add_objectvio_iter else self.loss(model_outputs, ground_truth,
                                                                             call_reg=False)
                # if change the pixel sampling pattern to patch, then you can add a TV loss to enforce some smoothness constraint
                loss = loss_output['loss']
                if 'sampling_loss' in model_outputs:
                    loss += model_outputs['sampling_loss']

                bg_loss = self.calculate_background_recon_loss(self.bg_info, local_model)
                
                loss += bg_loss
                loss_output['bg_loss'] = bg_loss

                loss.backward()

                local_optimizer.step()
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1, 3))
                iter_step += 1
                if iter_step % 20 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, beta={9}, alpha={10}, \n '
                        'semantic_loss = {11}, reg_loss = {12}, bg = {13}, bg_loss = {14} \n'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                psnr.item(),
                                local_model.density.get_beta().item(),
                                1. / local_model.density.get_beta().item(),
                                loss_output['semantic_loss'].item(),
                                loss_output['collision_reg_loss'].item(),
                                loss_output['background_reg_loss'].item(),
                                bg_loss.item(),
                                ))
                local_scheduler.step()
                if iter_step >= total_iterations:
                    break

            if iter_step >= total_iterations:
                break

        with torch.no_grad():
            mesh_extracted = marching_cubes_from_sdf(local_model.implicit_network, 0)
            if mesh_extracted.faces.shape[0] > 250000:
                mesh_extracted = simplify_mesh(mesh_extracted, 250000)
        
        mesh_extracted = generate_color_from_model_and_mesh(local_model, mesh_extracted, 0)
        
        with torch.no_grad():
            self.mesh_coarse_recon_dict[0] = mesh_extracted

            trimesh.exchange.export.export_mesh(
                mesh_extracted,
                os.path.join(self.plots_dir, f"coarse_recon_obj_0.ply")
            )

            self.mesh_coarse_points_collisions_dict[0] = {}
            for desc_obj_i in self.train_dataset.graph_node_dict[0]['desc']:
                x = np.linspace(-1, 1, collision_sample_cnt)
                y = np.linspace(-1, 1, collision_sample_cnt)
                z = np.linspace(-1, 1, collision_sample_cnt)
                points = np.stack(
                    np.meshgrid(x, y, z, indexing='ij'), axis=-1
                ).reshape(-1, 3)

                bbox_desc_obj_i = self.original_bbox_dict[desc_obj_i]
                center = bbox_desc_obj_i['center']
                scale = bbox_desc_obj_i['scale']

                points = points * float(scale) * collision_sample_scale_factor + center.reshape(1, 3)

                points = torch.from_numpy(points).float()

                sdfs = []
                for _pts in torch.split(points, 1024, dim=0):
                    sdfs.append(
                        local_model.implicit_network.get_sdf_raw(_pts.cuda())[:, 0].cpu())
                sdfs = torch.cat(sdfs, dim=0).reshape(-1)

                inside_pts = sdfs < 0
                self.mesh_coarse_points_collisions_dict[0][desc_obj_i] = {
                    'points': points.cpu().numpy(),
                    'sdfs': sdfs.cpu().numpy()
                }

            collision_pts_sdf_save_path = os.path.join(self.plots_dir, f"coarse_recon_obj_collision_pts_sdf_0.pkl")
            with open(collision_pts_sdf_save_path, 'wb') as f:
                pickle.dump(self.mesh_coarse_points_collisions_dict[0], f)

        local_model = local_model.cpu()

        self.train_dataset.sampling_class_id = -1

    def foreground_object_reconstruction(self, obj_i, view_dict_list, additional_pts_sdfs=None,
                            total_iterations=250, main_loss_weight=1.0, start_collision_loss_ratio=0., vis=False,
                            collision_sample_cnt=64, collision_sample_scale_factor=0.6):
        print("begin fg recon locally")
        bbox_obj_i = self.original_bbox_dict[obj_i]
        center = torch.from_numpy(bbox_obj_i['center']).reshape(-1, 3).float()
        scale = float(bbox_obj_i['scale'])
        conf_model = self.conf.get_config('model')
        local_model = utils.get_class(self.conf.get_string('train.model_class'))(
            conf=conf_model,
            plots_dir=self.plots_dir,
            graph_node_dict=self.train_dataset.graph_node_dict,
            ft_folder=self.ft_folder,
            num_images=len(self.train_dataset.mvps)
        )

        local_optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(local_model.implicit_network.grid_parameters()),
             'lr': self.lr * self.lr_factor_for_grid},
            {'name': 'net', 'params': list(local_model.implicit_network.mlp_parameters()) + \
                                      list(local_model.rendering_network.parameters()),
             'lr': self.lr},
            {'name': 'density', 'params': list(local_model.density.parameters()),
             'lr': self.lr},
        ], betas=(0.9, 0.99), eps=1e-15)

        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        local_scheduler = torch.optim.lr_scheduler.ExponentialLR(local_optimizer, decay_rate ** (1. / decay_steps))

        local_model = local_model.cuda()
        local_model.load_state_dict(copy.deepcopy(self.ckpt_dict['model']))
        local_model.train()


        

        self.train_dataset.sampling_class_id = obj_i
        self.train_dataset.change_sampling_idx(self.num_pixels)

        local_optimizer.load_state_dict(copy.deepcopy(self.ckpt_dict['optimizer']))
        local_scheduler.load_state_dict(copy.deepcopy(self.ckpt_dict['scheduler']))
        print("finish loading ckpt to local models")

        parent_sdf = None

        for obj_j in range(local_model.implicit_network.d_out):
            if obj_i != obj_j and obj_j in self.mesh_coarse_points_collisions_dict and obj_i in self.mesh_coarse_points_collisions_dict[obj_j]:
                print("collision obj_j: ", obj_j)
                if parent_sdf is None:
                    parent_sdf = self.mesh_coarse_points_collisions_dict[obj_j][obj_i]['sdfs']
                else:
                    parent_sdf = np.where(parent_sdf > self.mesh_coarse_points_collisions_dict[obj_j][obj_i]['sdfs'],
                                          self.mesh_coarse_points_collisions_dict[obj_j][obj_i]['sdfs'], parent_sdf)
        assert parent_sdf is not None
        parent_sdf = torch.from_numpy(parent_sdf).float()

        total_epoch = total_iterations // len(self.train_dataloader) + 1

        with torch.no_grad():
            x = np.linspace(-1, 1, collision_sample_cnt)
            y = np.linspace(-1, 1, collision_sample_cnt)
            z = np.linspace(-1, 1, collision_sample_cnt)
            points = np.stack(
                np.meshgrid(x, y, z, indexing='ij'), axis=-1
            ).reshape(-1, 3)

            bbox_obj_i = self.original_bbox_dict[obj_i]
            center = bbox_obj_i['center']
            scale = bbox_obj_i['scale']

            points = points * float(scale) * collision_sample_scale_factor + center.reshape(1, 3)

            points = torch.from_numpy(points).float()

            sdfs = []
            for _pts in torch.split(points, 1024, dim=0):
                sdfs.append(
                    local_model.implicit_network.get_sdf_raw(_pts.cuda())[:, obj_i].cpu())
            self_sdfs = torch.cat(sdfs, dim=0).reshape(-1)
            self_points = points

            
        self_points = self_points.reshape(collision_sample_cnt, collision_sample_cnt, collision_sample_cnt, 3)
        self_sdfs = self_sdfs.reshape(collision_sample_cnt, collision_sample_cnt, collision_sample_cnt)
        parent_sdf = parent_sdf.reshape(collision_sample_cnt, collision_sample_cnt, collision_sample_cnt)

        parent_intersection = parent_sdf < 0
        self_maintain = torch.logical_and(parent_sdf > 0, self_sdfs < 0)
        do_parent_intersection = torch.any(parent_intersection)
        do_self_maintain = torch.any(self_maintain)

        if do_parent_intersection:
            parent_intersection_idxs = torch.nonzero(parent_intersection).reshape(-1, 3)

        if do_self_maintain:
            self_maintain_idxs = torch.nonzero(self_maintain).reshape(-1, 3)
            self_maintain_sdf = torch.where(self_sdfs > -parent_sdf, self_sdfs, -parent_sdf)

        do_collision_loss = do_parent_intersection or do_self_maintain

        random_offset = float(scale) * collision_sample_scale_factor / collision_sample_cnt
        print("random_offset: ", random_offset)

        prune_base_threshold_list = [-4, -2, 0, 1, 2]
        base_tolarant_trans_list = [10, 8, 4, 4, 10000]

        mesh_extracted_init_list = marching_cubes_from_sdf_center_scale_rm_intersect(local_model.implicit_network, parent_sdf, center, float(scale),
                                                                           random_offset, collision_sample_scale_factor, collision_sample_cnt,
                                                                           obj_i, prune_base_threshold_list=prune_base_threshold_list)

        for mesh_extracted_init_i, mesh_extracted_init in enumerate(mesh_extracted_init_list):

            if mesh_extracted_init.faces.shape[0] > 25000:
                mesh_extracted_init = simplify_mesh(mesh_extracted_init, 25000)
            mesh_extracted_init = remesh(mesh_extracted_init)
            mesh_extracted_init = generate_color_from_model_and_mesh(local_model, mesh_extracted_init, obj_i)
            assert mesh_extracted_init is not None

            mesh_extracted_init_list[mesh_extracted_init_i] = [mesh_extracted_init, prune_base_threshold_list[mesh_extracted_init_i], random_offset, base_tolarant_trans_list[mesh_extracted_init_i]]

        iter_step = 0
        init_collision = True
        for epoch in range(total_epoch):
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                if vis and iter_step % 100 == 0:
                    data_idx = np.random.choice(len(view_dict_list))
                    print("length of len(view_dict_list): ", len(view_dict_list))
                    print("data_idx: ", data_idx)
                    gen_data_dict = view_dict_list[data_idx]
                    rgb = gen_data_dict["rgb"]
                    normal = gen_data_dict["normal"]
                    depth = gen_data_dict["depth"]
                    mask = gen_data_dict["mask"]
                    pose = gen_data_dict["pose"]
                    scale = gen_data_dict["scale"]
                    subset_idxs = gen_data_dict["obj_idxs"]
                    H, W = normal.shape[:2]
                    ray_origins, ray_dirs = get_camera_orthogonal_rays(H, W, near=0.001, pose=pose.cpu().clone(),
                                                   scale=scale)

                    normal_pred_list = []
                    mask_pred_list = []
                    rgb_pred_list = []
                    depth_pred_list = []
                    pose = pose.cuda()

                    for ray_o, ray_d in zip(torch.split(ray_origins, 1024, dim=0), torch.split(ray_dirs, 1024, dim=0)):
                        out = local_model.forward_multi_obj_rays_subset_all_sdf_near_far(ray_o.cuda(), ray_d.cuda(),
                                                                                         pose, subset_idxs, subset_idxs,
                                                                                         near=0.001, far=4.0*scale)
                        normal_pred = out['normal_map'].reshape(-1, 3).detach().cpu().numpy()
                        mask_pred = out['opacity'].reshape(-1).detach().cpu().numpy()
                        rgb_pred = out['rgb_values'].reshape(-1, 3).detach().cpu().numpy()
                        depth_pred = out['depth_values'].reshape(-1).detach().cpu().numpy()

                        normal_pred_list.append(normal_pred)
                        mask_pred_list.append(mask_pred)
                        rgb_pred_list.append(rgb_pred)
                        depth_pred_list.append(depth_pred)

                    normal_pred = np.concatenate(normal_pred_list, axis=0).reshape(H, W, 3)
                    mask_pred = np.concatenate(mask_pred_list, axis=0).reshape(H, W)
                    rgb_pred = np.concatenate(rgb_pred_list, axis=0).reshape(H, W, 3)
                    depth_pred = np.concatenate(depth_pred_list, axis=0).reshape(H, W)

                    depth_min = depth[mask].min()
                    depth_max = depth[mask].max()
                    depth_vis = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
                    depth_pred_vis = np.clip((depth_pred - depth_min) / (depth_max - depth_min), 0, 1)

                    image_label = f"iter_{iter_step:0>4d}"
                    Image.fromarray(np.clip((normal_pred * 0.5 + 0.5)*255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_normal_pred.png"))

                    Image.fromarray(np.clip((mask_pred) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_mask_pred.png"))

                    Image.fromarray(np.clip((rgb_pred) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_rgb_pred.png"))

                    Image.fromarray(np.clip((normal.reshape(H, W, 3) * 0.5 + 0.5) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_normal.png"))

                    Image.fromarray(np.clip((mask.reshape(H, W)) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_mask.png"))

                    Image.fromarray(np.clip((rgb) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_rgb.png"))

                    Image.fromarray(np.clip((depth_vis) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_depth.png"))

                    Image.fromarray(np.clip((depth_pred_vis) * 255., 0, 255).astype(np.uint8)).save(
                        os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_{image_label}_depth_pred.png"))


                # print("into dataloader")
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()

                local_optimizer.zero_grad()

                model_outputs = local_model(model_input, indices, iter_step=iter_step)
                model_outputs['iter_step'] = iter_step

                loss = torch.tensor(0.0).cuda().float()
                loss_output = {}

                loss_output = self.loss(model_outputs, ground_truth, call_reg=True) if \
                    self.iter_step >= self.add_objectvio_iter else self.loss(model_outputs, ground_truth,
                                                                             call_reg=False)
                # if change the pixel sampling pattern to patch, then you can add a TV loss to enforce some smoothness constraint
                loss = loss_output['loss'] * main_loss_weight
                if 'sampling_loss' in model_outputs:
                    loss += model_outputs['sampling_loss']

                if len(view_dict_list) > 0:
                    invis_angle_loss = self.calculate_invisible_loss(view_dict_list, local_model, near=0.001, far=4.0*scale)
                    loss += invis_angle_loss
                else:
                    invis_angle_loss = torch.tensor(0.0).cuda().float()
                loss_output['invis_angle_loss'] = invis_angle_loss

                loss_collision = torch.tensor(0.0).cuda().float()

                if do_collision_loss and iter_step >= start_collision_loss_ratio * total_iterations:
                # if False:
                    if init_collision or iter_step % 250 == 0:
                        init_collision = False

                        with torch.no_grad():
                            x = np.linspace(-1, 1, collision_sample_cnt)
                            y = np.linspace(-1, 1, collision_sample_cnt)
                            z = np.linspace(-1, 1, collision_sample_cnt)
                            init_points = np.stack(
                                np.meshgrid(x, y, z, indexing='ij'), axis=-1
                            ).reshape(-1, 3)

                            bbox_obj_i = self.original_bbox_dict[obj_i]
                            center = bbox_obj_i['center']
                            scale = bbox_obj_i['scale']

                            init_points = init_points * float(scale) * collision_sample_scale_factor + center.reshape(1, 3)

                            init_points = torch.from_numpy(init_points).float()

                            init_sdfs = []
                            for _pts in torch.split(init_points, 1024, dim=0):
                                init_sdfs.append(
                                    local_model.implicit_network.get_sdf_raw(_pts.cuda())[:, obj_i].cpu())
                            init_sdfs = torch.cat(init_sdfs, dim=0).reshape(-1)

                            init_sdfs = init_sdfs.reshape(collision_sample_cnt, collision_sample_cnt, collision_sample_cnt)

                            if do_parent_intersection:
                                # init_collision_pts_idx = init_sdfs[parent_intersection] < -parent_sdf[parent_intersection]
                                init_collision_pts_idx = torch.logical_and(init_sdfs < -parent_sdf, parent_intersection)
                                init_collision_pts_idx = torch.nonzero(init_collision_pts_idx).reshape(-1, 3).cpu()

                                print("finish collision init, init_collision_pts_idx: ", init_collision_pts_idx.shape)

                            if do_self_maintain:
                                init_self_maintain_pts_idx = torch.logical_and(init_sdfs > self_maintain_sdf, self_maintain)
                                init_self_maintain_pts_idx = torch.nonzero(init_self_maintain_pts_idx).reshape(-1, 3).cpu()



                    if do_parent_intersection:
                        
                        sampled_idx = np.random.choice(parent_intersection_idxs.shape[0], min(4096, parent_intersection_idxs.shape[0]), replace=False)
                        sampled_idx = torch.from_numpy(sampled_idx)
                        sampled_idx = parent_intersection_idxs[sampled_idx]
                        extra_sampled_idx = np.random.choice(init_collision_pts_idx.shape[0], min(4096, init_collision_pts_idx.shape[0]), replace=False)
                        extra_sampled_idx = torch.from_numpy(extra_sampled_idx)
                        extra_sampled_idx = init_collision_pts_idx[extra_sampled_idx]
                        sampled_idx = torch.cat([sampled_idx, extra_sampled_idx], dim=0)
                        sampled_idx = sampled_idx.float()

                        sampled_idx = sampled_idx + torch.rand_like(sampled_idx) - 0.5
                        collision_pts_sampled = grid_sample_3d_with_channels(self_points, sampled_idx).cuda()
                        collision_sdfs_sampled = grid_sample_3d(parent_sdf, sampled_idx).cuda()
                        
                        valid_pts = collision_sdfs_sampled < 0
                        collision_pts_sampled = collision_pts_sampled[valid_pts]
                        collision_sdfs_sampled = collision_sdfs_sampled[valid_pts]
                        
                        loss_collision += local_model.get_pts_sdf_contraints_loss(obj_i, collision_pts_sampled,
                                                                                  collision_sdfs_sampled)
                    if do_self_maintain > 0:

                        sampled_idx = np.random.choice(self_maintain_idxs.shape[0],
                                                       min(4096, self_maintain_idxs.shape[0]), replace=False)
                        sampled_idx = torch.from_numpy(sampled_idx)
                        sampled_idx = self_maintain_idxs[sampled_idx]

                        extra_sampled_idx = np.random.choice(init_self_maintain_pts_idx.shape[0],
                                                             min(4096, init_self_maintain_pts_idx.shape[0]), replace=False)
                        extra_sampled_idx = torch.from_numpy(extra_sampled_idx)
                        extra_sampled_idx = init_self_maintain_pts_idx[extra_sampled_idx]
                        sampled_idx = torch.cat([sampled_idx, extra_sampled_idx], dim=0)


                        sampled_idx = sampled_idx.float()

                        sampled_idx = sampled_idx + torch.rand_like(sampled_idx) - 0.5

                        collision_pts_sampled = grid_sample_3d_with_channels(self_points, sampled_idx).cuda()
                        parent_sdf_sampled = grid_sample_3d(parent_sdf, sampled_idx).cuda()
                        self_sdf_sampled = grid_sample_3d(self_sdfs, sampled_idx).cuda()
                        valid_pts = torch.logical_and(self_sdf_sampled < 0, parent_sdf_sampled > 0)
                        collision_pts_sampled = collision_pts_sampled[valid_pts]
                        collision_sdfs_sampled = torch.where(self_sdf_sampled > -parent_sdf_sampled, self_sdf_sampled, -parent_sdf_sampled)[valid_pts]
                        # print(f"collision_pts_sampled: {collision_pts_sampled.shape}")
                        # print(f"collision_sdfs_sampled: {collision_sdfs_sampled.shape}")

                        if collision_pts_sampled.shape[0] > 0 and collision_sdfs_sampled.shape[0] > 0:
                            loss_collision += local_model.get_pts_sdf_maintain_loss(obj_i, collision_pts_sampled,
                                                                                    collision_sdfs_sampled)
                loss += loss_collision
                loss_output['collision_loss'] = loss_collision

                loss.backward()

                local_optimizer.step()
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1, 3))
                iter_step += 1
                if iter_step % 20 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, beta={9}, alpha={10}, \n '
                        'semantic_loss = {11}, reg_loss = {12}, bg = {13}, \n '
                        'invis_loss = {14}, collision_loss = {15}'
                        .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                psnr.item(),
                                local_model.density.get_beta().item(),
                                1. / local_model.density.get_beta().item(),
                                loss_output['semantic_loss'].item(),
                                loss_output['collision_reg_loss'].item(),
                                loss_output['background_reg_loss'].item(),
                                invis_angle_loss.item(),
                                loss_output['collision_loss'].item(),
                                ))
                local_scheduler.step()

                if iter_step >= total_iterations:
                    break

            if iter_step >= total_iterations:
                break

        # mesh_extracted = marching_cubes_from_sdf_center_scale(local_model.implicit_network, center, scale, obj_i)
        
        mesh_extracted_list = marching_cubes_from_sdf_center_scale_rm_intersect(local_model.implicit_network, parent_sdf, center, float(scale),
                                                                           random_offset, collision_sample_scale_factor, collision_sample_cnt,
                                                                           obj_i, prune_base_threshold_list=prune_base_threshold_list)

        valid_recon_exist = False
        for mesh_extracted_i, mesh_extracted in enumerate(mesh_extracted_list):
            if mesh_extracted is None:
                mesh_extracted_list[mesh_extracted_i] = (
                None, prune_base_threshold_list[mesh_extracted_i], random_offset,
                base_tolarant_trans_list[mesh_extracted_i])
                continue

            valid_recon_exist = True

            if mesh_extracted.faces.shape[0] > 25000:
                mesh_extracted = simplify_mesh(mesh_extracted, 25000)
            mesh_extracted = remesh(mesh_extracted)
            mesh_extracted = generate_color_from_model_and_mesh(local_model, mesh_extracted, obj_i)

            mesh_extracted_list[mesh_extracted_i] = (mesh_extracted, prune_base_threshold_list[mesh_extracted_i], random_offset, base_tolarant_trans_list[mesh_extracted_i])

        # self.mesh_coarse_recon_dict[obj_i] = mesh_extracted_list

        # if not valid_recon_exist:
        #     mesh_extracted_list = mesh_extracted_init_list
        mesh_extracted_list = mesh_extracted_list + mesh_extracted_init_list
        
        with torch.no_grad():

            self.mesh_coarse_points_collisions_dict[obj_i] = {}
            for desc_obj_i in self.train_dataset.graph_node_dict[obj_i]['desc'] + self.train_dataset.graph_node_dict[obj_i]['brothers']:
                x = np.linspace(-1, 1, collision_sample_cnt)
                y = np.linspace(-1, 1, collision_sample_cnt)
                z = np.linspace(-1, 1, collision_sample_cnt)
                points = np.stack(
                    np.meshgrid(x, y, z, indexing='ij'), axis=-1
                ).reshape(-1, 3)

                bbox_desc_obj_i = self.original_bbox_dict[desc_obj_i]
                center = bbox_desc_obj_i['center']
                scale = bbox_desc_obj_i['scale']

                points = points * float(scale) * collision_sample_scale_factor + center.reshape(1, 3)

                points = torch.from_numpy(points).float()

                sdfs = []
                for _pts in torch.split(points, 1024, dim=0):
                    sdfs.append(
                        local_model.implicit_network.get_sdf_raw(_pts.cuda())[:, obj_i].cpu())
                sdfs = torch.cat(sdfs, dim=0).reshape(-1)


                # inside_pts = sdfs < 0
                self.mesh_coarse_points_collisions_dict[obj_i][desc_obj_i] = {
                    'points': points.cpu().numpy(),
                    'sdfs': sdfs.cpu().numpy()
                }

                # # save some visualization
                # # create a open3d point cloud
                # pcd = o3d.geometry.PointCloud()
                # # use points as the point cloud
                # pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
                # # for inside pts, use red color, for outside pts, use blue color
                # colors = np.zeros((points.shape[0], 3))
                # colors[inside_pts.cpu().numpy()] = [1, 0, 0]
                # colors[~inside_pts.cpu().numpy()] = [0, 0, 1]
                # pcd.colors = o3d.utility.Vector3dVector(colors)
                # # save the point cloud
                # o3d.io.write_point_cloud(os.path.join(self.plots_dir, f"coarse_recon_obj_collision_pts_sdf_{obj_i}_{desc_obj_i}.ply"), pcd)

            collision_pts_sdf_save_path = os.path.join(self.plots_dir, f"coarse_recon_obj_collision_pts_sdf_{obj_i}.pkl")
            with open(collision_pts_sdf_save_path, 'wb') as f:
                pickle.dump(self.mesh_coarse_points_collisions_dict[obj_i], f)

        local_model = local_model.cpu()

        # torch.save(
        #     {"epoch": self.start_epoch, "model_state_dict": local_model.state_dict()},
        #     os.path.join(self.checkpoints_path, self.model_params_subdir, f"{self.start_epoch}_{obj_i}.pth"))


        self.train_dataset.sampling_class_id = -1

        return mesh_extracted_list


    def sampling_views_around_coarse_recon(self, mesh_recon, texture_dict, all_mesh_dict, obj_i, best_azi, pose_scale, pose_shift, proj_scale):

        save_dir = os.path.join(self.plots_dir, f"coarse_recon_obj_{obj_i}_rendering")
        os.makedirs(save_dir, exist_ok=True)

        azimuths = np.linspace(0., 360., 30)
        # azimuths = np.linspace(best_azi - 60, best_azi + 60, 30)
        phis = np.linspace(90-30, 90+30, 5)

        cam_proj = get_camera_orthogonal_projection_matrix(near=0.001, far=10.0, scale=proj_scale)
        cam_proj = torch.from_numpy(cam_proj).float().cuda()

        vt = torch.from_numpy(texture_dict['vt']).cuda()
        ft = torch.from_numpy(texture_dict['ft']).cuda()
        texture_map = torch.from_numpy(texture_dict['texture_map']).cuda()

        mesh_dict = {
            'vertices': F.pad(
                torch.from_numpy(mesh_recon.vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(mesh_recon.faces).int().to("cuda").contiguous(),
            'vt': vt,
            'ft': ft,
            'texture_map': texture_map,
        }

        mesh_dict_raw = {
            'vertices': F.pad(
                torch.from_numpy(all_mesh_dict[obj_i].vertices).float().to("cuda").contiguous(),
                pad=(0, 1), value=1.0, mode='constant'),
            'pos_idx': torch.from_numpy(all_mesh_dict[obj_i].faces).int().to("cuda").contiguous(),
        }

        H = W = 256
        resolution = (H, W)

        face_normals = get_faces_normal(mesh_recon.vertices, mesh_recon.faces)
        face_normals = torch.from_numpy(face_normals).cuda().float()

        face_normals_pred = get_faces_normal(all_mesh_dict[obj_i].vertices, all_mesh_dict[obj_i].faces)
        face_normals_pred = torch.from_numpy(face_normals_pred).cuda().float()

        sampled_views = []
        # bg_color = np.array([1., 1., 1.]).astype(np.float32).reshape(3)
        bg_color = np.random.rand(3).astype(np.float32).reshape(3)

        for azimuth in tqdm(azimuths):
            for phi in phis:
                pose = build_camera_matrix_from_angles_and_locs(azimuth, phi, pose_scale, pose_shift)
                pose = pose.float().cuda()
                w2c = torch.inverse(pose)

                mvp = cam_proj @ torch.inverse(pose)

                valid, triangle_id, depth, rgb = rasterize_mesh_with_uv(mesh_dict, mvp, self.glctx, resolution, pose)
                normal = face_normals[triangle_id.reshape(-1)].reshape(H, W, 3)
                normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
                normal = normal @ w2c[:3, :3].T

                valid = valid.cpu().numpy().reshape(H, W, 1)
                normal = normal.cpu().numpy()
                depth = depth.cpu().numpy().reshape(H, W, 1)
                rgb = rgb.cpu().numpy().reshape(H, W, 3)

                rgb[~valid.reshape(H, W)] = np.array([1.,1.,1.]).astype(np.float32).reshape(3)


                mask_pred, triangle_id_pred, depth_pred = rasterize_mesh(mesh_dict_raw, mvp, self.glctx, resolution, pose)
                mask_pred = mask_pred.reshape(H, W).cpu().numpy()

                # normal_pred = face_normals_pred[triangle_id_pred.reshape(-1)].reshape(H, W, 3)
                # normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
                # normal_pred = normal_pred @ w2c[:3, :3].T
                #
                # normal_pred = normal_pred.cpu().numpy()
                # depth_pred = depth_pred.cpu().numpy().reshape(H, W, 1)
                #
                # normal[mask_pred] = normal_pred[mask_pred]
                # depth[mask_pred] = depth_pred[mask_pred]

                diff_mask = np.logical_xor(mask_pred, valid.reshape(H, W))
                diff_mask = np.logical_and(diff_mask, valid.reshape(H, W))
                # diff_mask = binary_dilation(diff_mask, iterations=3)

                image_label = f"azimuth_{int(azimuth):0>3d}_phi_{int(phi):0>3d}"
                depth_min = depth[valid].min()
                depth_max = depth[valid].max()
                depth_vis = (depth - depth_min) / (depth_max - depth_min + 1e-5)
                depth_vis = np.concatenate([depth_vis, depth_vis, depth_vis], axis=-1)

                Image.fromarray(
                    np.clip(np.concatenate([normal*0.5+0.5, valid], axis=-1)*255, 0, 255).astype(np.uint8)
                ).save(os.path.join(save_dir, f"{image_label}_normal.png"))
                Image.fromarray(
                    np.clip(np.concatenate([depth_vis, valid], axis=-1)*255, 0, 255).astype(np.uint8)
                ).save(os.path.join(save_dir, f"{image_label}_depth.png"))
                Image.fromarray(
                    np.clip(np.concatenate([rgb, valid], axis=-1)*255, 0, 255).astype(np.uint8)
                ).save(os.path.join(save_dir, f"{image_label}_rgb.png"))


                Image.fromarray(
                    np.clip(mask_pred * 255, 0, 255).astype(np.uint8)
                ).save(os.path.join(save_dir, f"{image_label}_mask_pred.png"))

                Image.fromarray(
                    np.clip(diff_mask * 255, 0, 255).astype(np.uint8)
                ).save(os.path.join(save_dir, f"{image_label}_diff_mask.png"))

                sampled_views.append({
                    'rgb': rgb,
                    'normal': normal,
                    'mask': valid.reshape(H, W),
                    "pose": pose.cpu(),
                    'scale': proj_scale,
                    'depth': depth.reshape(H, W),
                    "obj_idxs": [obj_i],
                    "front": False,
                    'source': "wonder3d",
                    "lambda": 1.0,
                    "bg_color": bg_color,
                    "diff_mask": diff_mask,
                    "lambda_mask" : 20.0,
                    "lambda_nm_l1" : 0.5,
                    "lambda_nm_cos" : 0.5,
                    "lambda_depth" : 10.0,
                    "lambda_rgb" : 1.,
                })

        return sampled_views

    def solve_intersection(self):
        recon_mesh_dict = self.mesh_coarse_recon_dict
        graph_node_dict = self.train_dataset.graph_node_dict
        num_objs = self.model.implicit_network.d_out

        sim_mesh_list = [recon_mesh_dict[0]]
        sim_mesh_dict = {}
        sim_mesh_dict[0] = recon_mesh_dict[0]
        translation_dict = {}

        objs_seq_with_distance = [(obj_i, graph_node_dict[obj_i]['dist_to_root']) for obj_i in range(1, num_objs)]
        # sort objs_seq_with_distance by distance
        objs_seq_with_distance.sort(key=lambda x: x[1])
        objs_seq = [obj_i[0] for obj_i in objs_seq_with_distance]

        for _i, obj_i in enumerate(objs_seq):
            mesh_obj_i = recon_mesh_dict[obj_i]
            mesh_obj_i = trimesh.Trimesh(vertices=mesh_obj_i.vertices, faces=mesh_obj_i.faces, process=False)
            parent_id = graph_node_dict[obj_i]['parent']
            if parent_id != 0:
                base_translation = translation_dict[parent_id]
            else:
                base_translation = np.zeros(3).astype(np.float32).reshape(3)

            mesh_obj_i.vertices = mesh_obj_i.vertices + base_translation.reshape(1, 3)

            edge_length = np.mean(np.linalg.norm(mesh_obj_i.vertices[mesh_obj_i.faces[:, 0]] -
                                                 mesh_obj_i.vertices[mesh_obj_i.faces[:, 1]],
                                                 axis=1))
            for _ in range(100):
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
            print("processing: ", f"{_i} / {len(objs_seq)}")

        save_sim_mesh_dir = os.path.join(self.plots_dir, f"coarse_recon_obj_sim_mesh")
        os.makedirs(save_sim_mesh_dir, exist_ok=True)

        for obj_i in range(num_objs):
            mesh_obj_i = sim_mesh_dict[obj_i]
            mesh_obj_i.export(os.path.join(save_sim_mesh_dir, f"coarse_recon_obj_{obj_i:0>3d}.ply"))

        save_translate_path = os.path.join(save_sim_mesh_dir, f"translation_dict.pkl")
        with open(save_translate_path, 'wb') as f:
            pickle.dump(translation_dict, f)

        save_graph_node_dict_path = os.path.join(save_sim_mesh_dir, f"graph_node_dict.pkl")
        with open(save_graph_node_dict_path, 'wb') as f:
            pickle.dump(graph_node_dict, f)


        return [sim_mesh_dict[obj_i] for obj_i in range(num_objs) if obj_i in sim_mesh_dict.keys()]

    def update_graph_node_dict(self, all_meshes, support_normal_threshold=0.90):
        """
        Create a scene graph from all_meshes and return graph_node_dict.
        
        Args:
            all_meshes (list): List of trimesh objects
            support_normal_threshold (float): Threshold for determining support relationships
            
        Returns:
            dict: Dictionary containing node properties with keys 'parent', 'root', 'leaf', 'layer', 'desc'
        """
        # Create scene graph from meshes (parent-child relationships)
        parent_dict, child_dict = create_scene_graph_from_meshes(all_meshes, support_normal_threshold)
        
        # Convert to adjacency list format like graph.json
        total_num_objs = len(all_meshes) - 1  # Excluding background
        graph = convert_parent_child_to_adjacency_list(parent_dict, child_dict, total_num_objs)
        
        # Extract graph node properties from adjacency list
        graph_node_dict = extract_graph_node_properties(graph)
        
        return graph_node_dict