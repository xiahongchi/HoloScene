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



def extract_graph_node_properties(graph):
    # Convert the adjacency list format to a dictionary for easier access
    adjacency_dict = defaultdict(set)
    for node in graph:
        node_id = node["node_id"]
        for adj in node["adj_nodes"]:
            adjacency_dict[node_id].add(adj)
            adjacency_dict[adj].add(node_id)

    n = len(graph)
    root = 0

    # Calculate parents and tree structure
    parents = {root: -1}
    tree = defaultdict(list)

    # BFS to determine parents and tree structure
    visited = set()
    queue = deque([(root, 0, None)])  # (node, depth, parent)

    while queue:
        node, depth, parent = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        # Set parent if not root
        if parent is not None and node != root:
            parents[node] = parent

        # Add children to tree
        if parent is not None:
            tree[parent].append(node)

        # Explore neighbors
        for neighbor in adjacency_dict[node]:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1, node))

    # Identify leaf nodes
    leaf_nodes = set()
    for node in range(n):
        if node not in tree or len(tree[node]) == 0:
            leaf_nodes.add(node)

    # Find all descendants for each node
    def find_all_descendants(node):
        descendants = []

        def dfs(curr_node):
            for child in tree.get(curr_node, []):
                descendants.append(child)
                dfs(child)

        dfs(node)
        return sorted(descendants)

    # Compute descendants for all nodes
    all_descendants = {node: find_all_descendants(node) for node in range(n)}

    # Determine batch sequence and layer
    batches = []
    remaining_nodes = set(range(n))
    layer_map = {}

    while remaining_nodes:
        # Find leaf nodes among remaining nodes
        current_batch = [
            node for node in remaining_nodes
            if node in leaf_nodes or
               (node not in tree and node in remaining_nodes) or
               (node in tree and not any(child in remaining_nodes for child in tree[node]))
        ]

        if not current_batch:
            break

        # Sort the batch for consistency
        current_batch = sorted(current_batch)
        batches.append(current_batch)

        # Assign layer
        for node in current_batch:
            layer_map[node] = len(batches) - 1

        # Remove these nodes from remaining nodes
        for node in current_batch:
            remaining_nodes.remove(node)

    # measure the distance of every node to the root
    dist_to_root_all = {}
    for obj_i in range(n):
        dist_to_root = 0
        _obj_i_cur = obj_i
        while _obj_i_cur != root:
            dist_to_root += 1
            _obj_i_cur = parents[_obj_i_cur]
        dist_to_root_all[obj_i] = dist_to_root

    # Construct the graph node dictionary
    graph_node_dict = {}
    for node in range(n):
        graph_node_dict[node] = {
            'parent': parents.get(node, -1),
            'root': node == root,
            'leaf': node in leaf_nodes,
            'layer': layer_map.get(node, -1),
            'desc': all_descendants[node],
            'dist_to_root': dist_to_root_all[node],
        }

    return graph_node_dict

class NSDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_root_dir,
                 data_dir,
                 img_res,
                 scene_normalize_scale=1.0,
                 test_split=False,
                 test_split_ratio=0.1,
                 prior_dir="",
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

        self.sampling_flag = False  
        self.sampling_size = 1024  

        image_dir = os.path.join(self.instance_dir, "images")
        image_paths = [os.path.join(image_dir, im_name) for im_name in sorted(os.listdir(image_dir))]

        depth_dir = os.path.join(self.instance_dir, prior_dir, "depth")
        depth_paths = [os.path.join(depth_dir, im_name) for im_name in sorted(os.listdir(depth_dir))]
        print("loading depth prior from: ", depth_dir)

        normal_dir = os.path.join(self.instance_dir, prior_dir, "normal")
        normal_paths = [os.path.join(normal_dir, im_name) for im_name in sorted(os.listdir(normal_dir))]
        print("loading normal prior from: ", normal_dir)

        instance_mask_dir = os.path.join(self.instance_dir, "instance_mask")
        instance_mask_paths = [os.path.join(instance_mask_dir, im_name) for im_name in sorted(os.listdir(instance_mask_dir))]

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

        self.scene_mesh_path = os.path.join(self.instance_dir, "mesh.ply")

        graph_path = os.path.join(self.instance_dir, "graph.json")
        if os.path.exists(graph_path):
            with open(graph_path, 'r') as f:
                graph_nodes = json.load(f)
            self.graph_node_dict = extract_graph_node_properties(graph_nodes)
            # visualize_graph_tree(self.graph_node_dict, os.path.join(self.instance_dir, "graph_vis"))
        else:
            self.graph_node_dict = None

        mask_paths = None

        self.n_images = len(image_paths)
        print('[INFO]: Dataset Size ', self.n_images)

        self.intrinsics_all = []

        camera_info_path = os.path.join(self.instance_dir, "transforms.json")
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)

        fx = camera_info['fl_x']
        fy = camera_info['fl_y']
        cx = camera_info['cx']
        cy = camera_info['cy']

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
        print("loading rgb images...")
        for path in tqdm(image_paths):
            rgb = np.array(Image.open(path)).astype(np.float32) / 255.
            rgb = rgb.transpose(2, 0, 1)
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

        self.semantic_images = []
        self.semantic_images_classes = []
        num_instances = 0
        class_id_occurences = {}

        print("loading instance masks...")
        for ins_i, ipath in tqdm(enumerate(instance_mask_paths), total=len(instance_mask_paths)):
            instance_mask_pic = Image.open(ipath)
            instance_mask = np.array(instance_mask_pic).astype(np.uint8).reshape(-1, 1)
            background = instance_mask == 255
            instance_mask += 1
            instance_mask[background] = 0

            num_instances = max(num_instances, instance_mask.max())
            classes_frame = torch.sort(torch.from_numpy(np.unique(instance_mask)).int())[0]
            self.semantic_images_classes.append(classes_frame)
            
            for obj_i in range(num_instances + 1):
                class_id_occurences[obj_i] = class_id_occurences.get(obj_i, [])
                if np.count_nonzero(instance_mask == obj_i) >= 8:
                    class_id_occurences[obj_i].append(ins_i)

            self.semantic_images.append(torch.from_numpy(instance_mask).float())

        self.class_id_occurences = class_id_occurences
        for obj_i in range(num_instances + 1):
            print("obj_i: ", obj_i, "len: ", len(class_id_occurences[obj_i]))

        self.label_mapping = list(range(num_instances+1))

        self.mask_images = []
        for depth in self.depth_images:
            mask = torch.ones_like(depth)
            self.mask_images.append(mask)

        self.sampling_class_id = -1

        self.test_split = test_split
        self.fix_length = fix_length

        if test_split:

            num_test_split = int(self.n_images * test_split_ratio)
            train_split_indices = np.linspace(0, self.n_images - 1, self.n_images - num_test_split).astype(np.int32)
            test_split_indices = np.setdiff1d(np.arange(self.n_images), train_split_indices).tolist()
            train_split_indices = train_split_indices.tolist()
            self.test_mvps = [data for idx, data in enumerate(self.mvps) if idx in test_split_indices]
            self.test_pose_all = [data for idx, data in enumerate(self.pose_all) if idx in test_split_indices]
            self.test_intrinsics_all = [data for idx, data in enumerate(self.intrinsics_all) if idx in test_split_indices]
            self.test_rgb_images = [data for idx, data in enumerate(self.rgb_images) if idx in test_split_indices]
            self.test_depth_images = [data for idx, data in enumerate(self.depth_images) if idx in test_split_indices]
            self.test_normal_images = [data for idx, data in enumerate(self.normal_images) if idx in test_split_indices]
            self.test_semantic_images = [data for idx, data in enumerate(self.semantic_images) if idx in test_split_indices]
            self.test_semantic_images_classes = [data for idx, data in enumerate(self.semantic_images_classes) if idx in test_split_indices]
            self.test_mask_images = [data for idx, data in enumerate(self.mask_images) if idx in test_split_indices]
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
            self.mask_images = [data for idx, data in enumerate(self.mask_images) if idx in train_split_indices]
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

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx],
            "segs": self.semantic_images[idx]
        }


        if self.sampling_flag:

            if self.sampling_class_id == -1:
                half_sampling_size = self.sampling_size // 2
                num_classes_frame = len(self.semantic_images_classes[idx])
                sample_per_sem = half_sampling_size // num_classes_frame
                sample_bg = half_sampling_size - sample_per_sem * (num_classes_frame - 1)
                self.sampling_idx = []
                for class_i in range(num_classes_frame):
                    sem_idx = self.semantic_images_classes[idx][class_i]
                    mask = self.semantic_images[idx] == sem_idx
                    mask = mask.reshape(-1)
                    mask = torch.nonzero(mask).reshape(-1)
                    if class_i == 0:
                        if len(mask) > sample_bg:
                            mask = mask[torch.randperm(len(mask))[:sample_bg]]
                    else:
                        if len(mask) > sample_per_sem:
                            mask = mask[torch.randperm(len(mask))[:sample_per_sem]]
                    self.sampling_idx.append(mask)
                self.sampling_idx.append(torch.randperm(self.total_pixels)[:self.sampling_size-half_sampling_size])
                self.sampling_idx = torch.cat(self.sampling_idx, dim=0)
            else:
                mask = self.semantic_images[idx] == self.sampling_class_id
                mask = mask.reshape(-1)
                mask = torch.nonzero(mask).reshape(-1)

                if len(mask) > self.sampling_size:
                    mask = mask[torch.randperm(len(mask))[:self.sampling_size]]

                self.sampling_idx = mask


            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
            ground_truth["segs"] = self.semantic_images[idx][self.sampling_idx, :]

            sample["uv"] = uv[self.sampling_idx, :]
            sample["is_patch"] = torch.tensor([False])
            sample['sampling_idx'] = self.sampling_idx

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_flag = False
        else:
            self.sampling_flag = True
            self.sampling_size = sampling_size
