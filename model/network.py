import os
import json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hashencoder.hashgrid import HashEncoder

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np
import math
import trimesh
from torch import vmap


class ObjectImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0, # radius of the sphere in geometric initialization
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True, # use hash grid embedding or not, if not, it is a pure MLP with sin/cos embedding
            sigmoid = 20,
            color_grid_feature = False,
    ):
        super().__init__()
        
        self.d_out = d_out
        self.sigmoid = sigmoid
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale

        self.color_grid_feature = color_grid_feature
        if not self.color_grid_feature:
            dims = [d_in] + dims + [d_out + feature_vector_size]
        else:
            dims = [d_in] + dims + [d_out]

        self.embed_fn = None
        self.divide_factor = divide_factor
        self.use_grid_feature = use_grid_feature


        
        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                    per_level_scale=2, base_resolution=base_size,
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        # self.encoding = tcnn.Encoding(3, {
        #     "otype": "HashGrid",
        #     "n_levels": num_levels,  # Use the same value as in your original code
        #     "n_features_per_level": level_dim,  # Use the same value as in your original code
        #     "log2_hashmap_size": logmap,  # Use the same value as in your original code
        #     "base_resolution": base_size,  # Use the same value as in your original code
        #     "per_level_scale": 2.0,  # Use the same value as in your original code (2.0)
        #     "interpolation": "Linear"  # New parameter
        # })

        self.grid_feature_dim = num_levels * level_dim
        # self.grid_feature_dim = self.encoding.n_output_dims
        dims[0] += self.grid_feature_dim

        if self.color_grid_feature:
            self.color_encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                    per_level_scale=2, base_resolution=base_size,
                    log2_hashmap_size=logmap, desired_resolution=end_size)
            # self.color_encoding = tcnn.Encoding(3, {
            #     "otype": "HashGrid",
            #     "n_levels": num_levels,  # Use the same value as in your original code
            #     "n_features_per_level": level_dim,  # Use the same value as in your original code
            #     "log2_hashmap_size": logmap,  # Use the same value as in your original code
            #     "base_resolution": base_size,  # Use the same value as in your original code
            #     "per_level_scale": 2.0,  # Use the same value as in your original code (2.0)
            #     "interpolation": "Linear"  # New parameter
            # })
            self.color_grid_feature_dim = num_levels * level_dim
            # self.color_grid_feature_dim = self.color_encoding.n_output_dims

            # create a mlp map from self.color_grid_feature_dim to feature_vector_size
            self.color_grid_feature_map_mlp = nn.Sequential(
                nn.Linear(self.color_grid_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, feature_vector_size)
            )
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        self.encoding = tcnn.Encoding(3, {
                            "otype": "HashGrid",
                            "n_levels": num_levels,  # Use the same value as in your original code
                            "n_features_per_level": level_dim,  # Use the same value as in your original code
                            "log2_hashmap_size": logmap,  # Use the same value as in your original code
                            "base_resolution": base_size,  # Use the same value as in your original code
                            "per_level_scale": 2.0,  # Use the same value as in your original code (2.0)
                            "interpolation": "Linear"  # New parameter
                        })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        # print("network architecture")
        # print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias)
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.5 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.5*bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

        self.pool = nn.MaxPool1d(self.d_out, return_indices=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            # assert torch.max(input / self.divide_factor)<1 and torch.min(input / self.divide_factor)>-1, 'range out of [-1, 1], max: {}, min: {}'.format(torch.max(input / self.divide_factor),  torch.min(input / self.divide_factor))
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))

        if self.color_grid_feature:
            color_feature = self.color_encoding(input / self.divide_factor)
            color_feature = self.color_grid_feature_map_mlp(color_feature)

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            # debug
            # print(f"Input shape: {x.shape}")
            # print(f"Input contains NaN: {torch.isnan(x).any()}")
            # print(f"Input contains Inf: {torch.isinf(x).any()}")
            # # Check the linear layer weights
            # print(f"Weight norm: {lin.weight_v.norm()}")  
            # # for weight_norm layers
            # print(f"Weight contains NaN: {torch.isnan(lin.weight_v).any()}")
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        if self.color_grid_feature:
            x = torch.cat([x, color_feature], dim=-1)
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        f = lambda v: torch.autograd.grad(outputs=y,
                    inputs=x,
                    grad_outputs=v.repeat(y.shape[0], 1),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        
        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)
        
        # start_time = time.time()
        if self.use_grid_feature: # using hashing grid feature, cannot support vmap now
            g = torch.cat([torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=idx.repeat(y.shape[0], 1),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0] for idx in N.unbind()])
        # torch.cuda.synchronize()
        # print("time for computing gradient by for loop: ", time.time() - start_time, "s")
                
        # using vmap for batched gradient computation, if not using grid feature (pure MLP)
        else:
            g = vmap(f, in_dims=1)(N).reshape(-1, 3)
        
        # add the gradient of scene sdf
        # sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-y.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def gradient_obj_i(self, x, obj_i):
        x.requires_grad_(True)
        y = self.forward(x)[:, obj_i:obj_i+1]

        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)

        # start_time = time.time()
        g = torch.cat([torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=idx.repeat(y.shape[0], 1),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0] for idx in N.unbind()])

        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))

        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            # change semantic to the gradianct of density
            semantic = 1/beta * (0.5 + 0.5 * sdf_raw.sign() * torch.expm1(-sdf_raw.abs() / beta))
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw



    def get_sdf_vals(self, x):
        sdf_raw = self.forward(x)[:,:self.d_out]
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf

    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]
    
    def get_object_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, idx]
        return sdf

    def get_multi_object_sdf_vals(self, x, idxs):
        sdf_raw = self.forward(x)[:, idxs]
        pool = nn.MaxPool1d(len(idxs), return_indices=True)
        sdf, indices = pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)

        return sdf

    def get_sdf_vals_and_sdfs(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        sdf_raw = sdf
        # sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf, sdf_raw

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw[:, idx]

    def get_multi_specific_outputs(self, x, idxs):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        sdf_raw_idxs = sdf_raw[:, idxs]
        pool = nn.MaxPool1d(len(idxs), return_indices=True)
        sdf_specific, _ = pool(-sdf_raw_idxs.unsqueeze(1))
        sdf_specific = -sdf_specific.squeeze(-1)

        return sdf, feature_vectors, gradients, semantic, sdf_specific


    def get_only_multi_specific_outputs(self, x, idxs):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:, idxs]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        pool = nn.MaxPool1d(len(idxs), return_indices=True)
        sdf, indices = pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic

    def get_multi_specific_outputs_subset_objs(self, x, idxs, subset_idxs):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        sdf_raw_subset = output[:, subset_idxs]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw_subset)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        pool_subset = nn.MaxPool1d(len(subset_idxs), return_indices=True)
        sdf, indices = pool_subset(-sdf_raw_subset.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        sdf_raw_idxs = sdf_raw[:, idxs]
        pool = nn.MaxPool1d(len(idxs), return_indices=True)
        sdf_specific, _ = pool(-sdf_raw_idxs.unsqueeze(1))
        sdf_specific = -sdf_specific.squeeze(-1)

        return sdf, feature_vectors, gradients, semantic, sdf_specific

    def get_specific_outputs_nm(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        # sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        # sdf = -sdf.squeeze(-1)
        sdf = sdf_raw[:, idx]
        # indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic
    
    def get_shift_sdf_raw(self, x):
        sdf_raw = self.forward(x)[:, :self.d_out]
        # print("sdf_raw: ", sdf_raw.shape)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        # print("sdf: ", sdf.shape, sdf.max(), sdf.min())
        # print("indices: ", indices.shape)

        # shift raw sdf
        pos_min_sdf = -sdf          # other object sdf must bigger than -sdf
        # print("pos_min_sdf: ", pos_min_sdf.shape, pos_min_sdf.max(), pos_min_sdf.min())
        pos_min_sdf_expand = pos_min_sdf.expand_as(sdf_raw)
        shift_mask = (sdf < 0)
        shift_mask_expand = shift_mask.expand_as(sdf_raw)

        shift_sdf_raw = torch.where(shift_mask_expand, torch.max(sdf_raw, pos_min_sdf_expand), sdf_raw)
        shift_sdf_raw[torch.arange(indices.size(0)), indices.squeeze()] = sdf.squeeze()

        return shift_sdf_raw


    def get_outputs_and_indices(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]

        # if self.sigmoid_optim:
        #     sigmoid_value = torch.exp(self.sigmoid_basis)
        # else:
        sigmoid_value = self.sigmoid

        semantic = sigmoid_value * torch.sigmoid(-sigmoid_value * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, sdf_raw, indices

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        if self.color_grid_feature:
            parameters += list(self.color_grid_feature_map_mlp.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        verbose = False
        if self.color_grid_feature:
            if verbose:
                print("[INFO]: grid parameters", len(list(self.encoding.parameters())) + len(list(self.color_encoding.parameters())))
                for p in self.encoding.parameters():
                    print(p.shape)
                for p in self.color_encoding.parameters():
                    print(p.shape)
            return list(self.encoding.parameters()) + list(self.color_encoding.parameters())
        else:
            if verbose:
                print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
                for p in self.encoding.parameters():
                    print(p.shape)
            return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_point=0,
            multires_normal=0,
            num_images=1024
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        self.multires_view = multires_view
        self.multires_point = multires_point
        self.multires_normal = multires_normal
        if multires_view > 0 or multires_point > 0 or multires_normal > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn

            if multires_view > 0:
                dims[0] += (input_ch - 3)

            if multires_point > 0 and self.mode == 'idr':
                dims[0] += (input_ch - 3)

            if multires_normal > 0 and self.mode == 'idr':
                dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.multires_view > 0:
            view_dirs = self.embedview_fn(view_dirs)

        if self.multires_point > 0:
            points = self.embedview_fn(points)

        if self.multires_normal > 0:
            normals = self.embedview_fn(normals)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        # x.shape [N_rays*pts_per_ray, 7]
        color = self.sigmoid(x[:, :3])
        return color


class ColorImplicitNetwork(nn.Module):
    def __init__(
            self,
            base_size=16,
            end_size=2048,
            logmap=19,
            num_levels=16,
            level_dim=2,
            divide_factor=1.5,  # used to normalize the points range for multi-res grid
            obj_emb_len=32,
            num_objs=None
    ):
        super().__init__()
        self.grid_feature_dim = num_levels * level_dim
        self.divide_factor = divide_factor

        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                                    per_level_scale=2, base_resolution=base_size,
                                    log2_hashmap_size=logmap, desired_resolution=end_size)
        # self.encoding = tcnn.Encoding(3, {
        #     "otype": "HashGrid",
        #     "n_levels": num_levels,  # Use the same value as in your original code
        #     "n_features_per_level": level_dim,  # Use the same value as in your original code
        #     "log2_hashmap_size": logmap,  # Use the same value as in your original code
        #     "base_resolution": base_size,  # Use the same value as in your original code
        #     "per_level_scale": 2.0,  # Use the same value as in your original code (2.0)
        #     "interpolation": "Linear"  # New parameter
        # })

        assert num_objs is not None, 'num_objs should be set'

        self.embeddings = nn.Parameter(torch.empty(num_objs, obj_emb_len))
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

        self.color_mlp = nn.Sequential(
            nn.Linear(self.grid_feature_dim+obj_emb_len, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, input, obj_indices):
        feature = self.encoding(input / self.divide_factor)
        rgb = self.color_mlp(
            torch.cat([feature, self.embeddings[obj_indices]], dim=-1)
        )
        rgb = torch.nn.functional.sigmoid(rgb)

        return rgb

    def mlp_parameters(self):
        parameters = []
        parameters += list(self.color_mlp.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        verbose = False

        if verbose:
            print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
            for p in self.encoding.parameters():
                print(p.shape)
        return self.encoding.parameters()

class ColorImplicitNetworkSingle(nn.Module):
    def __init__(
            self,
            base_size=16,
            end_size=2048,
            logmap=19,
            num_levels=16,
            level_dim=2,
            divide_factor=1.5,  # used to normalize the points range for multi-res grid
    ):
        super().__init__()
        self.grid_feature_dim = num_levels * level_dim
        self.divide_factor = divide_factor

        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                                    per_level_scale=2, base_resolution=base_size,
                                    log2_hashmap_size=logmap, desired_resolution=end_size)
        # self.encoding = tcnn.Encoding(3, {
        #     "otype": "HashGrid",
        #     "n_levels": num_levels,  # Use the same value as in your original code
        #     "n_features_per_level": level_dim,  # Use the same value as in your original code
        #     "log2_hashmap_size": logmap,  # Use the same value as in your original code
        #     "base_resolution": base_size,  # Use the same value as in your original code
        #     "per_level_scale": 2.0,  # Use the same value as in your original code (2.0)
        #     "interpolation": "Linear"  # New parameter
        # })


        self.color_mlp = nn.Sequential(
            nn.Linear(self.grid_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, input):
        feature = self.encoding(input / self.divide_factor)
        rgb = self.color_mlp(feature)
        rgb = torch.nn.functional.sigmoid(rgb)

        return rgb

    def mlp_parameters(self):
        parameters = []
        parameters += list(self.color_mlp.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        verbose = False

        if verbose:
            print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
            for p in self.encoding.parameters():
                print(p.shape)
        return self.encoding.parameters()

class HoloSceneNetwork(nn.Module):
    def __init__(self,
                  conf,
                  plots_dir=None,
                  graph_node_dict=None,
                  ft_folder=None,
                  num_images=1024,
                  ):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.use_bg_reg = conf.get_bool('use_bg_reg', default=False)
        self.render_bg_iter = conf.get_int('render_bg_iter', default=10)
        self.graph_node_dict = graph_node_dict

        self.implicit_network = ObjectImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        self.num_semantic = conf.get_int('implicit_network.d_out')
        self.rendering_network = RenderingNetwork(self.feature_vector_size, num_images=num_images, **conf.get_config('rendering_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        

        self.plots_dir = plots_dir
        self.ft_folder = ft_folder              # if continue training

        self.all_mesh_bbox_dict = None

    def forward(self, input, indices, iter_step=-1):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        if self.training:
            ray_offset = torch.rand_like(uv) - 0.5
        else:
            ray_offset = None
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics, ray_offset=ray_offset)
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics, ray_offset=ray_offset)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat, beta=None)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'semantic_values': semantic_values, # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity, 
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels 
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
            
            grad_theta = self.implicit_network.gradient(eikonal_points)

            if self.all_mesh_bbox_dict is None:

                sample_sdf = self.implicit_network.get_sdf_raw(eikonal_points)
                sdf_value = self.implicit_network.get_sdf_vals(eikonal_points)
                output['sample_sdf'] = sample_sdf
                output['sample_minsdf'] = sdf_value

            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]

        if self.training and self.all_mesh_bbox_dict is not None:
            n_eik_points = batch_size * num_pixels

            num_objs = self.implicit_network.d_out
            assert num_objs == self.collision_sampling_prob.shape[0], 'collision_sampling_prob shape should be equal to num_objs' + " num_objs: " + str(num_objs) + " collision_sampling_prob shape: " + str(self.collision_sampling_prob.shape)
            obj_idx = np.random.choice(num_objs, p=self.collision_sampling_prob)

            mesh_bbox_dict = self.all_mesh_bbox_dict[obj_idx]
            mesh_bbox_center = mesh_bbox_dict['center']
            mesh_bbox_scale = mesh_bbox_dict['scale']

            intersect_pts = mesh_bbox_dict['intersect_pts']
            # choose at most 4096 pts from torch tensor intersect_pts with shape (n, 3)
            sample_intersect_n_pts = intersect_pts.shape[0]
            sample_intersect_n_pts = min(sample_intersect_n_pts, 4096)
            sample_idxs = np.random.choice(intersect_pts.shape[0], sample_intersect_n_pts, replace=False)
            intersect_pts = intersect_pts[torch.from_numpy(sample_idxs).long().reshape(-1)]
            intersect_pts = intersect_pts.cuda()

            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-1.0, 1.0).cuda()
            eikonal_points = torch.cat([eikonal_points, intersect_pts], dim=0)

            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)

            eikonal_points = eikonal_points * torch.from_numpy(mesh_bbox_scale).float().cuda().reshape(1, 3) + torch.from_numpy(mesh_bbox_center).float().cuda().reshape(1, 3)

            sample_sdf = self.implicit_network.get_sdf_raw(eikonal_points)
            
            output['sample_sdf'] = sample_sdf
            output['collision_relations'] = mesh_bbox_dict['collision_relations']

            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = torch.cat([output['grad_theta'], grad_theta[:grad_theta.shape[0]//2]], 0)
            output['grad_theta_nei'] = torch.cat([output['grad_theta_nei'], grad_theta[grad_theta.shape[0]//2:]], 0)

        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        # only for render the background depth and normal
        iter_check = iter_step % self.render_bg_iter
        if self.use_bg_reg and iter_check == 0:

            # construct patch uv
            patch_size = 32
            n_patches = 1

            cx_2 = float(intrinsics[:, 0, 2].reshape(-1)[0]) * 2.0
            cy_2 = float(intrinsics[:, 1, 2].reshape(-1)[0]) * 2.0

            x0 = np.random.randint(0, int(cx_2) - patch_size + 1, size=(n_patches, 1, 1))         # NOTE: fix image resolution as 384
            y0 = np.random.randint(0, int(cy_2) - patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_idx = xy0 + np.stack(np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),axis=-1).reshape(1, -1, 2)
            uv0 = torch.from_numpy(patch_idx).float().reshape(1, -1, 2).float().cuda()
            ray_dirs0, cam_loc0 = rend_util.get_camera_params(uv0, pose, intrinsics)

            # we should use unnormalized ray direction for depth
            ray_dirs0_tmp, _ = rend_util.get_camera_params(uv0, torch.eye(4).to(pose.device)[None], intrinsics)
            depth_scale0 = ray_dirs0_tmp[0, :, 2:]
            
            batch_size, num_pixels, _ = ray_dirs0.shape

            cam_loc0 = cam_loc0.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
            ray_dirs0 = ray_dirs0.reshape(-1, 3)

            bg_z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs0, cam_loc0, self, idx=0)
            N_samples_bg = bg_z_vals.shape[1]

            bg_points = cam_loc0.unsqueeze(1) + bg_z_vals.unsqueeze(2) * ray_dirs0.unsqueeze(1)
            bg_points_flat = bg_points.reshape(-1, 3)
            scene_sdf, _, bg_gradients, scene_semantic, bg_sdf = self.implicit_network.get_specific_outputs(bg_points_flat, 0)
            
            bg_weight, _, _ = self.volume_rendering(bg_z_vals, bg_sdf)

            # NOTE: semantic should use scene sdf for volume rendering
            scene_weight, _, _ = self.volume_rendering(bg_z_vals, scene_sdf)
            scene_semantic = scene_semantic.reshape(-1, N_samples_bg, self.num_semantic)
            bg_semantic_value = torch.sum(scene_weight.unsqueeze(-1)*scene_semantic, 1)
            bg_mask = torch.argmax(bg_semantic_value, dim=-1, keepdim=True)
            output['bg_mask'] = bg_mask

            bg_depth_values = torch.sum(bg_weight * bg_z_vals, 1, keepdims=True) / (bg_weight.sum(dim=1, keepdims=True) +1e-8)
            bg_depth_values = depth_scale0 * bg_depth_values 
            output['bg_depth_values'] = bg_depth_values

            # compute bg normal map
            bg_normals = bg_gradients / (bg_gradients.norm(2, -1, keepdim=True) + 1e-6)
            bg_normals = bg_normals.reshape(-1, N_samples_bg, 3)
            bg_normal_map = torch.sum(bg_weight.unsqueeze(-1) * bg_normals, 1)
            bg_normal_map = rot @ bg_normal_map.permute(1, 0)
            bg_normal_map = bg_normal_map.permute(1, 0).contiguous()
            output['bg_normal_map'] = bg_normal_map


        return output

    def get_pts_sdf_contraints_loss(self, obj_i, points, sdfs):
        points = points.reshape(-1, 3)
        sdfs = sdfs.reshape(-1)
        sample_sdf = self.implicit_network.get_sdf_raw(points)[:, obj_i].reshape(-1)
        grad_theta = self.implicit_network.gradient_obj_i(points, obj_i)

        delta_sdf = -sample_sdf - sdfs
        collision_mask = delta_sdf > 0
        if torch.any(collision_mask):
            loss_sdf = torch.mean(delta_sdf[collision_mask])
        else:
            loss_sdf = torch.tensor(0.0).float().cuda()
        loss_eikonal = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()

        return loss_sdf * 5.0 + loss_eikonal * 0.1

    def get_pts_sdf_maintain_loss(self, obj_i, points, sdfs):
        points = points.reshape(-1, 3)
        sdfs = sdfs.reshape(-1)
        sample_sdf = self.implicit_network.get_sdf_raw(points)[:, obj_i].reshape(-1)
        grad_theta = self.implicit_network.gradient_obj_i(points, obj_i)

        delta_sdf = sample_sdf - sdfs
        collision_mask = delta_sdf > 0
        if torch.any(collision_mask):
            loss_sdf = torch.mean(delta_sdf[collision_mask])
        else:
            loss_sdf = torch.tensor(0.0).float().cuda()
        loss_eikonal = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return loss_sdf * 3.0 + loss_eikonal * 0.1

    def get_additional_sdf_loss(self, obj_i, points, sdfs):
        points = points.reshape(-1, 3)
        sdfs = sdfs.reshape(-1)
        sample_sdf = self.implicit_network.get_sdf_raw(points)[:, obj_i].reshape(-1)
        grad_theta = self.implicit_network.gradient_obj_i(points, obj_i)

        loss_sdf = torch.mean(torch.abs(sdfs - sample_sdf))
        loss_eikonal = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()

        return loss_sdf * 10.0 + loss_eikonal * 0.1


    def forward_multi_obj(self, input, indices, obj_idxs, iter_step=-1):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs(points_flat, idxs=obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors,
                                                               indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        bg_weights, _, _ = self.volume_rendering(z_vals, sdf_raw)

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(bg_weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(bg_weights * z_vals, 1, keepdims=True) / (bg_weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values
        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'bg_weights': bg_weights,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(bg_weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output


    def forward_multi_obj_rays(self, ray_origins, ray_dirs, pose, obj_idxs, iter_step=-1, sem_bg_weights=False):
        # Parse model input

        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        rot = pose[..., :3, :3].reshape(3, 3).permute(1, 0).contiguous()
        ray_dirs_cam = (rot @ ray_dirs.permute(1, 0)).permute(1, 0)
        depth_scale = ray_dirs_cam[:, 2:]

        cam_loc = ray_origins
        ray_dirs = ray_dirs

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs(points_flat, idxs=obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        bg_weights, _, _ = self.volume_rendering(z_vals, sdf_raw)

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(bg_weights.unsqueeze(-1) * rgb, 1)
        if sem_bg_weights:
            semantic_values = torch.sum(bg_weights.unsqueeze(-1) * semantic, 1)
        else:
            semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(bg_weights * z_vals, 1, keepdims=True) / (bg_weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values

        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'bg_weights': bg_weights,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(bg_weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output

    def forward_only_multi_obj_rays(self, ray_origins, ray_dirs, pose, obj_idxs, iter_step=-1):
        # Parse model input

        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        rot = pose[..., :3, :3].reshape(3, 3).permute(1, 0).contiguous()
        ray_dirs_cam = (rot @ ray_dirs.permute(1, 0)).permute(1, 0)
        depth_scale = ray_dirs_cam[:, 2:]

        cam_loc = ray_origins
        ray_dirs = ray_dirs

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic = self.implicit_network.get_only_multi_specific_outputs(points_flat, idxs=obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, len(obj_idxs))

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values

        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output

    def forward_multi_obj_rays_subset_all_sdf(self, ray_origins, ray_dirs, pose, obj_idxs, subset_obj_idxs, iter_step=-1):
        # Parse model input

        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        rot = pose[..., :3, :3].reshape(3, 3).permute(1, 0).contiguous()
        ray_dirs_cam = (rot @ ray_dirs.permute(1, 0)).permute(1, 0)
        depth_scale = ray_dirs_cam[:, 2:]

        cam_loc = ray_origins
        ray_dirs = ray_dirs

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=obj_idxs, subset_idxs=subset_obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)

        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, len(subset_obj_idxs))

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        bg_weights, _, _ = self.volume_rendering(z_vals, sdf_raw)

        # rendering the occlusion-awared object opacity
        opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(bg_weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(bg_weights * z_vals, 1, keepdims=True) / (bg_weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values

        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'opacity': opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'bg_weights': bg_weights,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(bg_weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output

    def forward_multi_obj_rays_subset_all_sdf_near_far(self, ray_origins, ray_dirs, pose, obj_idxs, subset_obj_idxs, near, far, iter_step=-1):
        # Parse model input

        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        rot = pose[..., :3, :3].reshape(3, 3).permute(1, 0).contiguous()
        ray_dirs_cam = (rot @ ray_dirs.permute(1, 0)).permute(1, 0)
        depth_scale = ray_dirs_cam[:, 2:]

        cam_loc = ray_origins
        ray_dirs = ray_dirs

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals_near_far(ray_dirs, cam_loc, self, near, far, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=obj_idxs, subset_idxs=subset_obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)

        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, len(subset_obj_idxs))

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        bg_weights, _, _ = self.volume_rendering(z_vals, sdf_raw)

        # rendering the occlusion-awared object opacity
        opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(bg_weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(bg_weights * z_vals, 1, keepdims=True)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values

        opacity = torch.sum(bg_weights, dim=-1).reshape(-1)


        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'opacity': opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'bg_weights': bg_weights,
        }


        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(bg_weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output


    def forward_multi_obj_rays_subset_all_sdf_detach_rgb_for_geometry(self, ray_origins, ray_dirs, pose, obj_idxs, subset_obj_idxs, iter_step=-1):
        # Parse model input

        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        rot = pose[..., :3, :3].reshape(3, 3).permute(1, 0).contiguous()
        ray_dirs_cam = (rot @ ray_dirs.permute(1, 0)).permute(1, 0)
        depth_scale = ray_dirs_cam[:, 2:]

        cam_loc = ray_origins
        ray_dirs = ray_dirs

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=obj_idxs, subset_idxs=subset_obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients.detach(), dirs_flat, feature_vectors, 0)

        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, len(subset_obj_idxs))

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        bg_weights, _, _ = self.volume_rendering(z_vals, sdf_raw)

        # rendering the occlusion-awared object opacity
        opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(bg_weights.unsqueeze(-1).detach() * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(bg_weights * z_vals, 1, keepdims=True) / (bg_weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values


        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'opacity': opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'bg_weights': bg_weights,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(bg_weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output


    def forward_multi_obj_rays_subset_all_sdf_detach_rgb_for_geometry_near_far(self, ray_origins, ray_dirs, pose, obj_idxs, subset_obj_idxs, near, far, iter_step=-1):
        # Parse model input

        ray_origins = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        rot = pose[..., :3, :3].reshape(3, 3).permute(1, 0).contiguous()
        ray_dirs_cam = (rot @ ray_dirs.permute(1, 0)).permute(1, 0)
        depth_scale = ray_dirs_cam[:, 2:]

        cam_loc = ray_origins
        ray_dirs = ray_dirs

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals_near_far(ray_dirs, cam_loc, self, near, far, idx=obj_idxs)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=obj_idxs, subset_idxs=subset_obj_idxs)

        rgb_flat = self.rendering_network(points_flat, gradients.detach(), dirs_flat, feature_vectors, 0)

        rgb = rgb_flat.reshape(-1, N_samples, 3)

        semantic = semantic.reshape(-1, N_samples, len(subset_obj_idxs))

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        bg_weights, _, _ = self.volume_rendering(z_vals, sdf_raw)

        # rendering the occlusion-awared object opacity
        opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(bg_weights.unsqueeze(-1).detach() * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1) * semantic, 1)
        depth_values = torch.sum(bg_weights * z_vals, 1, keepdims=True) / (bg_weights.sum(dim=1, keepdims=True) + 1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values


        output = {
            'rgb': rgb,
            'semantic_values': semantic_values,  # here semantic value calculated as in ObjectSDF
            'opacity': opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'bg_weights': bg_weights,
        }

        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(bg_weights.unsqueeze(-1) * normals, 1)

        # transform to local coordinate system
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        output['normal_map'] = normal_map

        return output


    def get_colors_normals_from_point_rays(self, points, rays, pose):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat, beta=None)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        rot_c2w = pose.reshape(-1, 4)[:3, :3]
        rot = rot_c2w.permute(1, 0).contiguous()

        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        return rgb_values, normal_map

    def get_colors_normals_from_point_rays_obj(self, points, rays, pose, obj_idx):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idx)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic = self.implicit_network.get_specific_outputs_nm(points_flat, idx=obj_idx)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        rot_c2w = pose.reshape(-1, 4)[:3, :3]
        rot = rot_c2w.permute(1, 0).contiguous()

        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)
        semantic_values = torch.argmax(semantic_values, dim=-1)

        return rgb_values, normal_map, semantic_values

    def get_colors_normals_from_point_rays_obj_f(self, points, rays, pose, obj_idx):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=obj_idx)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic = self.implicit_network.get_specific_outputs_nm(points_flat, idx=obj_idx)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        rot_c2w = pose.reshape(-1, 4)[:3, :3]
        rot = rot_c2w.permute(1, 0).contiguous()

        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)

        return rgb_values, normal_map, semantic_values

    def get_colors_from_point_rays(self, points, rays):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat, beta=None)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        return rgb_values

    def get_colors_from_point_rays_obj(self, points, rays, obj_i):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=[obj_i])
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=[obj_i], subset_idxs=[obj_i])

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        return rgb_values

    def get_colors_from_point_rays_obj_offset(self, points, rays, obj_i):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=[obj_i])
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=[obj_i], subset_idxs=[obj_i])

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        return rgb_values

    def get_colors_from_point_rays_obj_offset_near_far(self, points, rays, obj_i, near, far):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals_near_far(ray_dirs, cam_loc, self, idx=[obj_i], near=near, far=far)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=[obj_i], subset_idxs=[obj_i])

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        return rgb_values

    def get_colors_from_point_rays_obj_debug(self, points, rays, obj_i):
        points = points.reshape(-1, 3)
        rays = rays.reshape(-1, 3)
        rays = F.normalize(rays, dim=-1)

        cam_loc = points
        ray_dirs = rays

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, idx=[obj_i])
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_multi_specific_outputs_subset_objs(points_flat, idxs=[obj_i], subset_idxs=[obj_i])

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, 0)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_values = rgb_values.reshape(-1, 3)

        ray_weights = torch.sum(weights, 1)

        return rgb_values, ray_weights

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, transmittance, dists

    def occlusion_opacity(self, z_vals, transmittance, dists, sdf_raw):
        obj_density = self.density(sdf_raw).transpose(0, 1).reshape(-1, dists.shape[0], dists.shape[1]) # [#object, #ray, #sample points]       
        free_energy = dists * obj_density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        object_weight = alpha * transmittance
        return object_weight
 
    def get_parameters(self):
        # delete the parameters in uncertainty field
        params = []
        for name, param in self.named_parameters():
            if 'uncertainty_field' not in name:
                params.append(param)
        
        return params

class SingleObjectImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size=256,
            d_in = 3,
            d_out = 1,
            dims = [256, 256],
            geometric_init=True,
            bias=0.9, # radius of the sphere in geometric initialization
            skip_in=[4],
            weight_norm=True,
            multires=6,
            sphere_scale=1.0,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.0, # used to normalize the points range for multi-res grid
            use_grid_feature = True, # use hash grid embedding or not, if not, it is a pure MLP with sin/cos embedding
            sigmoid = 10,
            object_center=None,
            object_scale=None,
            fg_bg=True,
    ):
        super().__init__()

        self.d_out = d_out
        self.sigmoid = sigmoid
        self.sphere_scale = sphere_scale

        dims = [d_in] + dims + [d_out+feature_vector_size]

        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim

        self.object_center = object_center
        self.object_scale = object_scale

        self.fg_bg = fg_bg

        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                                    per_level_scale=2, base_resolution=base_size,
                                    log2_hashmap_size=logmap, desired_resolution=end_size)
        # self.encoding = tcnn.Encoding(3, {
        #     "otype": "HashGrid",
        #     "n_levels": num_levels,  # Use the same value as in your original code
        #     "n_features_per_level": level_dim,  # Use the same value as in your original code
        #     "log2_hashmap_size": logmap,  # Use the same value as in your original code
        #     "base_resolution": base_size,  # Use the same value as in your original code
        #     "per_level_scale": 2.0,  # Use the same value as in your original code (2.0)
        #     "interpolation": "Linear"  # New parameter
        # })

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    if not self.fg_bg:
                        torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias[:1], bias)
                    else:
                        # inner objects with SDF initial with negative value inside and positive value outside, ~0.5 radius of background
                        torch.nn.init.normal_(lin.weight[:1,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias[:1], -0.5*bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

        self.relu = nn.ReLU()

    def forward(self, input):

        feature = self.encoding((input - self.object_center) / self.object_scale / self.divide_factor)

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        f = lambda v: torch.autograd.grad(outputs=y,
                                          inputs=x,
                                          grad_outputs=v.repeat(y.shape[0], 1),
                                          create_graph=True,
                                          retain_graph=True,
                                          only_inputs=True)[0]

        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)

        g = torch.cat([torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=idx.repeat(y.shape[0], 1),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0] for idx in N.unbind()])

        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf = output[:,:self.d_out]

        feature_vectors = output[:, self.d_out:]

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:self.d_out]

        return sdf

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        verbose = False

        if verbose:
            print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
            for p in self.encoding.parameters():
                print(p.shape)
        return self.encoding.parameters()


class SingleObjectRenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size=256,
            mode='idr',
            d_in=9,
            d_out=3,
            dims=[256, 256],
            weight_norm=True,
            multires_view=4,
            multires_point=0,
            multires_normal=0,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        self.multires_view = multires_view
        self.multires_point = multires_point
        self.multires_normal = multires_normal
        if multires_view > 0 or multires_point > 0 or multires_normal > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn

            if multires_view > 0:
                dims[0] += (input_ch - 3)

            if multires_point > 0 and self.mode == 'idr':
                dims[0] += (input_ch - 3)

            if multires_normal > 0 and self.mode == 'idr':
                dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors,):
        if self.multires_view > 0:
            view_dirs = self.embedview_fn(view_dirs)

        if self.multires_point > 0:
            points = self.embedview_fn(points)

        if self.multires_normal > 0:
            normals = self.embedview_fn(normals)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # x.shape [N_rays*pts_per_ray, 7]
        color = self.sigmoid(x[:, :3])
        return color

class ObjectSDFNetwork(nn.Module):
    def __init__(self,
                 center,
                 scale,
                 fg_bg,
                 conf):
        super().__init__()

        self.scene_bounding_sphere = 1.0
        self.implicit_network = SingleObjectImplicitNetworkGrid(
            object_center=center,
            object_scale=scale,
            fg_bg=fg_bg,
        )

        self.rendering_network = SingleObjectRenderingNetwork()

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, transmittance, dists

    def occlusion_opacity(self, z_vals, transmittance, dists, sdf_raw):
        obj_density = self.density(sdf_raw).transpose(0, 1).reshape(-1, dists.shape[0], dists.shape[1]) # [#object, #ray, #sample points]
        free_energy = dists * obj_density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        object_weight = alpha * transmittance
        return object_weight

    def forward(self, ray_origins, ray_dirs):

        cam_loc = ray_origins.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat, beta=None)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)

        rgb = rgb_flat.reshape(-1, N_samples, 3)
        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)

        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf).sum(-1).transpose(0, 1)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)


        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        n_eik_points = 2048

        eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere,
                                                               self.scene_bounding_sphere).cuda()

        # add some of the near surface points
        eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
        eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
        # add some neighbour points as unisurf
        neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01
        eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)

        grad_theta = self.implicit_network.gradient(eikonal_points)

        output = {
            'object_opacity': object_opacity,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'normal_map': normal_map,
            'opacity': object_opacity
        }

        output['grad_theta'] = grad_theta[:grad_theta.shape[0] // 2]
        output['grad_theta_nei'] = grad_theta[grad_theta.shape[0] // 2:]

        return output