import os
import torch
import torch.nn.functional as F
import numpy as np
from utils.general import get_camera_perspective_projection_matrix, visualize_graph_tree

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
import json
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, deque


class NSDatasetTex(torch.utils.data.Dataset):

    def __init__(self,
                 data_root_dir,
                 data_dir,
                 img_res,
                 scene_normalize_scale=1.0,
                 test_split=False,
                 test_split_ratio=0.1,
                 fix_length=0,
                 max_num_images=-1
                 ):

        self.instance_dir = os.path.join(data_root_dir, data_dir)
        print(self.instance_dir)

        self.scene_normalize_scale = scene_normalize_scale

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        self.sampling_flag = False  # flag to indicate whether to sample the image (not sample image when inference)
        self.sampling_size = 1024  # default sampling size (include physics-guided sampling and random sampling)

        # read image paths
        image_dir = os.path.join(self.instance_dir, "images")
        image_paths = [os.path.join(image_dir, im_name) for im_name in sorted(os.listdir(image_dir))]

        depth_dir = os.path.join(self.instance_dir, "depth")
        depth_paths = [os.path.join(depth_dir, im_name) for im_name in sorted(os.listdir(depth_dir))]

        normal_dir = os.path.join(self.instance_dir, "normal")
        normal_paths = [os.path.join(normal_dir, im_name) for im_name in sorted(os.listdir(normal_dir))]

        instance_mask_dir = os.path.join(self.instance_dir, "instance_mask")
        instance_mask_paths = [os.path.join(instance_mask_dir, im_name) for im_name in sorted(os.listdir(instance_mask_dir))]
        self.scene_mesh_path = os.path.join(self.instance_dir, "mesh.ply")

        # Determine downsampling indices early if needed
        original_n_images = len(image_paths)
        if max_num_images > 0 and max_num_images < original_n_images:
            print(f"[INFO]: Downsampling from {original_n_images} to {max_num_images} images")
            downsample_indices = np.linspace(0, original_n_images - 1, max_num_images).astype(np.int32)
            
            # Filter paths to only load selected images
            image_paths = [image_paths[i] for i in downsample_indices]
            depth_paths = [depth_paths[i] for i in downsample_indices]
            normal_paths = [normal_paths[i] for i in downsample_indices]
            instance_mask_paths = [instance_mask_paths[i] for i in downsample_indices]
        else:
            downsample_indices = None

        self.n_images = len(image_paths)
        print('[INFO]: Dataset Size ', self.n_images)

        self.intrinsics_all = []

        camera_info_path = os.path.join(self.instance_dir, "transforms.json")
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)

        self.fx = fx = camera_info['fl_x']
        self.fy = fy = camera_info['fl_y']
        self.cx = cx = camera_info['cx']
        self.cy = cy = camera_info['cy']

        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        poses = []
        if downsample_indices is not None:
            # Only load camera data for downsampled indices
            for idx in downsample_indices:
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                pose = np.array(camera_info['frames'][idx]['transform_matrix']).reshape(4, 4)
                pose[:3, 1:3] *= -1
                poses.append(pose)
        else:
            # Load all camera data
            for idx in range(self.n_images):
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                pose = np.array(camera_info['frames'][idx]['transform_matrix']).reshape(4, 4)
                pose[:3, 1:3] *= -1
                poses.append(pose)

        poses = np.stack(poses, axis=0)
        # find the x, y, z-max min of the scene
        max_xyz = np.max(poses[..., :3, 3], axis=0)
        min_xyz = np.min(poses[..., :3, 3], axis=0)
        scene_center = (max_xyz + min_xyz) / 2
        scene_scale = np.max(max_xyz - min_xyz) * self.scene_normalize_scale

        poses[..., :3, 3] = (poses[..., :3, 3] - scene_center) / scene_scale

        self.scene_center = scene_center
        self.scene_scale = scene_scale

        poses_gl = poses.copy()
        poses_gl[:, :3, 1:3] *= -1

        n = 0.001
        f = 100  # infinite

        camera_projmat = get_camera_perspective_projection_matrix(fx, fy, cx, cy, img_res[0], img_res[1], n, f)

        # Update n_images to reflect actual loaded images
        self.n_images = len(poses)
        mvps = camera_projmat[None, ...].repeat(self.n_images, axis=0) @ np.linalg.inv(poses)
        self.mvps = [torch.from_numpy(mvp).float() for mvp in mvps]

        _poses = []
        for p in poses:
            # p = np.linalg.inv(p)
            # p[:3, :3] = p[:3, :3].transpose()
            _poses.append(torch.from_numpy(p).float())

        self.pose_all = _poses

        self.rgb_images = []
        for path in tqdm(image_paths):
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.depth_images = []
        self.normal_images = []

        print("loading depth and normal priors...")
        for dpath, npath in tqdm(zip(depth_paths, normal_paths), total=len(depth_paths)):
            # print(f"loading priors ({len(self.normal_images)}/{self.n_images}): ", dpath, npath)
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())

            normal = Image.open(npath)
            normal = np.array(normal).astype(np.float32) / 255.0
            normal = normal.reshape(-1, 3)
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        num_instances = 0
        self.semantic_images = []
        self.semantic_images_classes = []
        self.class_id_occurences = {}

        for ins_i, ipath in tqdm(enumerate(instance_mask_paths), total=len(instance_mask_paths)):
            instance_mask_pic = Image.open(ipath)
            instance_mask = np.array(instance_mask_pic).astype(np.uint8).reshape(-1, 1)
            background = instance_mask == 255
            instance_mask += 1
            instance_mask[background] = 0

            num_instances = max(num_instances, instance_mask.max())
            classes_frame = torch.sort(torch.from_numpy(np.unique(instance_mask)).int())[0]
            self.semantic_images_classes.append(classes_frame)
            self.semantic_images.append(torch.from_numpy(instance_mask).float())

            for obj_i in range(num_instances + 1):
                self.class_id_occurences[obj_i] = self.class_id_occurences.get(obj_i, [])
                if np.count_nonzero(instance_mask == obj_i) >= 8:
                    self.class_id_occurences[obj_i].append(ins_i)

        for obj_i in range(num_instances + 1):
            print("obj_i: ", obj_i, "len: ", len(self.class_id_occurences[obj_i]))
        self.label_mapping = list(range(num_instances+1))

        self.sampling_class_id = -1
        self.test_split = test_split
        self.fix_length = fix_length

        if test_split:

            num_test_split = int(self.n_images * test_split_ratio)
            train_split_indices = np.linspace(0, self.n_images - 1, self.n_images - num_test_split).astype(np.int32)
            test_split_indices = np.setdiff1d(np.arange(self.n_images), train_split_indices).tolist()
            train_split_indices = train_split_indices.tolist()
            print("self.n_images: ", self.n_images)
            # print("train_split_indices: ", len(train_split_indices), train_split_indices)
            # print("test_split_indices: ", len(test_split_indices), test_split_indices)
            self.test_mvps = [data for idx, data in enumerate(self.mvps) if idx in test_split_indices]
            self.test_pose_all = [data for idx, data in enumerate(self.pose_all) if idx in test_split_indices]
            self.test_intrinsics_all = [data for idx, data in enumerate(self.intrinsics_all) if idx in test_split_indices]
            self.test_rgb_images = [data for idx, data in enumerate(self.rgb_images) if idx in test_split_indices]
            self.test_depth_images = [data for idx, data in enumerate(self.depth_images) if idx in test_split_indices]
            self.test_normal_images = [data for idx, data in enumerate(self.normal_images) if idx in test_split_indices]
            self.test_semantic_images = [data for idx, data in enumerate(self.semantic_images) if idx in test_split_indices]
            self.test_semantic_images_classes = [data for idx, data in enumerate(self.semantic_images_classes) if idx in test_split_indices]
            self.test_class_id_occurences = {}
            for obj_i in range(num_instances + 1):
                self.test_class_id_occurences[obj_i] = []
                for test_i, data_idx in enumerate(test_split_indices):
                    if data_idx in self.class_id_occurences[obj_i]:
                        self.test_class_id_occurences[obj_i].append(test_i)

            self.mvps = [data for idx, data in enumerate(self.mvps) if idx in train_split_indices]
            self.pose_all = [data for idx, data in enumerate(self.pose_all) if idx in train_split_indices]
            self.intrinsics_all = [data for idx, data in enumerate(self.intrinsics_all) if idx in train_split_indices]
            self.rgb_images = [data for idx, data in enumerate(self.rgb_images) if idx in train_split_indices]
            self.depth_images = [data for idx, data in enumerate(self.depth_images) if idx in train_split_indices]
            self.normal_images = [data for idx, data in enumerate(self.normal_images) if idx in train_split_indices]
            self.semantic_images = [data for idx, data in enumerate(self.semantic_images) if idx in train_split_indices]
            self.semantic_images_classes = [data for idx, data in enumerate(self.semantic_images_classes) if idx in train_split_indices]
            self.train_class_id_occurences = {}
            for obj_i in range(num_instances + 1):
                self.train_class_id_occurences[obj_i] = []
                for train_i, data_idx in enumerate(train_split_indices):
                    if data_idx in self.class_id_occurences[obj_i]:
                        self.train_class_id_occurences[obj_i].append(train_i)
            self.class_id_occurences = self.train_class_id_occurences
            self.n_images = len(self.rgb_images)

            print("self.n_images: ", self.n_images)
            print("len (self.rgb_images): ", len(self.rgb_images))

    def __len__(self):
        return self.n_images if self.fix_length==0 else self.fix_length

    def __getitem__(self, idx):

        if self.fix_length != 0:
            idx = random.randint(0, self.n_images - 1)

        if self.sampling_class_id != -1:
            idx = np.random.choice(self.class_id_occurences[self.sampling_class_id], 1)[0]
            idx = int(idx)
            assert self.sampling_class_id in self.semantic_images_classes[idx]

        sample = {
            "intrinsics": [self.fx, self.fy, self.cx, self.cy],
            "pose": self.pose_all[idx],
            "image_res": self.img_res,
            "near_far": [0.001, 100.0],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "mask": self.semantic_images[idx],
            'depth': self.depth_images[idx],
        }

        return idx, sample, ground_truth
