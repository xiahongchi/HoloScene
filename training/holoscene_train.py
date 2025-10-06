import imp
import os
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
from model.network import HoloSceneNetwork
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import (
    get_time, refined_obj_bbox, get_lcc_mesh, get_scale_shift, apply_scale_shift,
    sample_views_around_object, apply_inv_scale_shift, build_camera_matrix, get_camera_orthogonal_rays,
    vis_prune, sample_views_around_object_backface, sample_views_around_object_naive, visualize_view_weights,
    get_camera_orthogonal_projection_matrix, rasterize_mesh_list, get_cam_normal_from_rast, get_faces_normal, get_camera_perspective_rays,
    find_best_additional_view, evaluate_view_addition, margin_aware_fps_sampling, highest_sampling_view_weights,
    find_largest_connected_region, build_camera_matrix_from_angles_and_locs, rasterize_mesh_list_front_face, resize_int_tensor, rasterize_mesh,
    get_camera_perspective_rays_world, rasterize_mesh_vert_colors
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
from scipy.ndimage import binary_dilation
from segment_anything import sam_model_registry, SamPredictor
from glob import glob
import open3d as o3d

class HoloSceneTrainRunner():
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

        dataset_conf['data_root_dir'] = os.path.abspath(dataset_conf['data_root_dir'])

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.stop_iter = self.conf.get_int('train.stop_iter', default=self.max_total_iters)
        self.ds_len = len(self.train_dataset)
        # self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        self.nepochs = int(self.max_total_iters / self.ds_len)
        self.stop_epoch = int(self.stop_iter / self.ds_len)
        print('RUNNING FOR {0} EPOCHS'.format(self.stop_epoch))

        if len(self.train_dataset.label_mapping) > 0:
            # a hack way to let network know how many categories, so don't need to manually set in config file
            self.conf['model']['implicit_network']['d_out'] = len(self.train_dataset.label_mapping)
            print('RUNNING FOR {0} CLASSES'.format(len(self.train_dataset.label_mapping)))

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
        decay_steps = self.nepochs * self.ds_len
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        single_gpu_mode = True

        self.start_epoch = 0
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')
            self.finetune_folder = os.path.join(self.expdir, timestamp)

            print('[INFO]: Loading pretrained model from {}'.format(old_checkpnts_dir))
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            model_state_dict = {}
            if single_gpu_mode:
                for k, v in saved_model_state["model_state_dict"].items():
                    model_state_dict[k.replace('module.', '')] = v
            else:
                model_state_dict = saved_model_state["model_state_dict"]
            self.model.load_state_dict(model_state_dict)
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            # continue training need copy mesh files from old folder
            old_plots_folder = os.path.join(self.finetune_folder, 'plots')
            mesh_str = f'surface_{self.start_epoch}_*'
            cmd = f'cp {old_plots_folder}/{mesh_str} {self.plots_dir}'
            os.system(cmd)
            cmd = f'cp -r {old_plots_folder}/bbox {self.plots_dir}'
            os.system(cmd)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()
        
        self.add_objectvio_iter = self.conf.get_int('train.add_objectvio_iter', default=100000)


        self.n_sem = self.conf.get_int('model.implicit_network.d_out')
        assert self.n_sem == len(self.train_dataset.label_mapping)

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

        self.iter_step_start = self.iter_step = self.start_epoch * self.ds_len
        print(f'Start epoch: {self.start_epoch}, iter_step: {self.iter_step}')
        for epoch in range(self.start_epoch, self.stop_epoch + 1):

            if epoch % self.checkpoint_freq == 0 and epoch != 0:
                self.save_checkpoints(epoch)

            if epoch % self.plot_freq == 0 and epoch != 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s, indices)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach(),
                         'depth_values': out['depth_values'].detach(),
                        }
                    if 'semantic_values' in out:
                        d['semantic_values'] = torch.argmax(out['semantic_values'].detach(),dim=1)
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'])

                obj_bbox_dict = None
                if os.path.exists(os.path.join(self.plots_dir, 'bbox')):        # use object bbox
                    obj_bbox_dict = {}
                    obj_list = os.listdir(os.path.join(self.plots_dir, 'bbox'))
                    for obj in obj_list:
                        obj_idx = int((obj.split('.')[0]).split('_')[1])
                        with open(os.path.join(self.plots_dir, 'bbox', obj), 'r') as f:
                            bbox = json.load(f)
                        obj_bbox_dict[obj_idx] = bbox
                
                plt.plot_color_mesh(self.model.implicit_network,
                                    self.model,
                                    indices,
                                    plot_data,
                                    self.plots_dir,
                                    epoch,
                                    self.iter_step,  # iter
                                    self.img_res,
                                    **self.plot_conf,
                                    obj_bbox_dict=obj_bbox_dict,
                                    level=0.002,
                                    plot_mesh=epoch > self.start_epoch,
                                    shift=True,
                                    trainer=self,
                                    color_mesh=False
                                    )

                all_meshes = self.instance_meshes_post_pruning(epoch)
                self.generate_bbox(all_meshes)

                self.model.train()
            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                self.optimizer.zero_grad()

                # visulize sampling pixels
                if self.iter_step % 1000 == 0:
                    vis_uv = model_input['uv'].squeeze(0).cpu().detach().numpy()        # [N_rays, 2]
                    vis_uv = vis_uv.astype(np.int32)
                    vis_rgb = ground_truth['full_rgb'].clone().squeeze(0).cpu().detach().numpy()     # [N_rays, 3]
                    vis_rgb = vis_rgb.reshape(self.img_res[0], self.img_res[1], 3)
                    vis_rgb = (vis_rgb * 255.0).astype(np.uint8)
                    vis_rgb[vis_uv[:, 1], vis_uv[:, 0], :] = [255, 0, 0]                # add red mask
                    vis_rgb = Image.fromarray(vis_rgb)
                    # create vis_pixels dir
                    vis_pixels_dir = os.path.join(self.plots_dir, 'vis_pixels')
                    os.makedirs(vis_pixels_dir, exist_ok=True)
                    vis_pixels_path = os.path.join(vis_pixels_dir, f'{epoch}_{indices[0]}.png')
                    vis_rgb.save(vis_pixels_path)
                
                model_outputs = self.model(model_input, indices, iter_step=self.iter_step)
                model_outputs['iter_step'] = self.iter_step
                
                loss_output = self.loss(model_outputs, ground_truth, call_reg=True) if\
                        self.iter_step >= self.add_objectvio_iter else self.loss(model_outputs, ground_truth, call_reg=False)
                # if change the pixel sampling pattern to patch, then you can add a TV loss to enforce some smoothness constraint
                loss = loss_output['loss']
                if 'sampling_loss' in model_outputs:
                    loss += model_outputs['sampling_loss']
                loss.backward()

                # calculate gradient norm
                total_norm = 0
                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                self.optimizer.step()
                
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                # .module
                self.iter_step += 1                
                
                if data_index %20 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, beta={9}, alpha={10}, semantic_loss = {11}, reg_loss = {12}, bg = {13}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.density.get_beta().item(),
                                    1. / self.model.density.get_beta().item(),
                                    loss_output['semantic_loss'].item(),
                                    loss_output['collision_reg_loss'].item(),
                                    loss_output['background_reg_loss'].item()
                                ))
                    
                    if self.use_wandb:
                        for k, v in loss_output.items():
                            wandb.log({f'Loss/{k}': v.item()}, self.iter_step)

                        if 'sampling_loss' in model_outputs:
                            wandb.log({'Loss/sampling_loss': model_outputs['sampling_loss'].item()}, self.iter_step)

                        wandb.log({'Statistics/beta': self.model.density.get_beta().item()}, self.iter_step)
                        wandb.log({'Statistics/alpha': 1. / self.model.density.get_beta().item()}, self.iter_step)
                        wandb.log({'Statistics/psnr': psnr.item()}, self.iter_step)
                        wandb.log({'Statistics/total_norm': total_norm}, self.iter_step)
                        
                        wandb.log({'Statistics/lr0': self.optimizer.param_groups[0]['lr']}, self.iter_step)
                        wandb.log({'Statistics/lr1': self.optimizer.param_groups[1]['lr']}, self.iter_step)
                        wandb.log({'Statistics/lr2': self.optimizer.param_groups[2]['lr']}, self.iter_step)

                    else:
                        for k, v in loss_output.items():
                            tb_writer.add_scalar(f'Loss/{k}', v.item(), self.iter_step)

                        if 'sampling_loss' in model_outputs:
                            tb_writer.add_scalar('Loss/sampling_loss', model_outputs['sampling_loss'].item(), self.iter_step)

                        tb_writer.add_scalar('Statistics/beta', self.model.density.get_beta().item(), self.iter_step)
                        tb_writer.add_scalar('Statistics/alpha', 1. / self.model.density.get_beta().item(), self.iter_step)
                        tb_writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                        tb_writer.add_scalar('Statistics/total_norm', total_norm, self.iter_step)

                        tb_writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                        tb_writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                        tb_writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

        self.save_checkpoints(epoch)

        print('********** finish export mesh **********')
        self.model.eval()

        self.train_dataset.change_sampling_idx(-1)

        indices, model_input, ground_truth = next(iter(self.plot_dataloader))
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        
        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
        res = []
        for s in tqdm(split):
            out = self.model(s, indices)
            d = {'rgb_values': out['rgb_values'].detach(),
                'normal_map': out['normal_map'].detach(),
                'depth_values': out['depth_values'].detach(),
                }
            if 'semantic_values' in out:
                d['semantic_values'] = torch.argmax(out['semantic_values'].detach(),dim=1)
            res.append(d)

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
        plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'])

        obj_bbox_dict = None
        if os.path.exists(os.path.join(self.plots_dir, 'bbox')):        # use object bbox
            obj_bbox_dict = {}
            obj_list = os.listdir(os.path.join(self.plots_dir, 'bbox'))
            for obj in obj_list:
                obj_idx = int((obj.split('.')[0]).split('_')[1])
                with open(os.path.join(self.plots_dir, 'bbox', obj), 'r') as f:
                    bbox = json.load(f)
                obj_bbox_dict[obj_idx] = bbox
        
        plt.plot_color_mesh(self.model.implicit_network,
                            self.model,
                            indices,
                            plot_data,
                            self.plots_dir,
                            epoch,
                            self.iter_step,  # iter
                            self.img_res,
                            **self.plot_conf,
                            obj_bbox_dict=obj_bbox_dict,
                            level=0.002,
                            plot_mesh=epoch > self.start_epoch,
                            shift=True,
                            trainer=self,
                            color_mesh=True
                            )

        all_meshes = self.instance_meshes_post_pruning(epoch)
        self.generate_bbox(all_meshes)

        if self.use_wandb:
            wandb.finish()
        print('training over')

   
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


    def instance_meshes_post_pruning(self, epoch, min_visible_verts=0):
        all_meshes = []
        faces_cnts = [0]
        num_objs = self.model.implicit_network.d_out
        for obj_i in range(num_objs):
            obj_i_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_{obj_i}.ply')
            assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
            obj_i_mesh = trimesh.exchange.load.load_mesh(obj_i_mesh_path)
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

            # face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
            faces_lcc[:, 0] = filter_unmapping[faces_lcc[:, 0]]
            faces_lcc[:, 1] = filter_unmapping[faces_lcc[:, 1]]
            faces_lcc[:, 2] = filter_unmapping[faces_lcc[:, 2]]

            vert_colors = obj_i_mesh.visual.vertex_colors[..., :3].reshape(-1, 3)
            vert_colors = vert_colors[verts_map]

            all_meshes[obj_i] = trimesh.Trimesh(vertices=verts_lcc, faces=faces_lcc, vertex_colors=vert_colors, process=False)

        # save back mesh
        for obj_i in range(num_objs):
            obj_i_mesh = all_meshes[obj_i]
            obj_i_mesh.export(os.path.join(self.plots_dir, f'surface_{epoch}_{obj_i}.ply'))

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
            # print(f'bbox_{mesh_i}.json save to {bbox_root_path}')
    

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
