#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from torch import nn
from einops import repeat
from utils.system_utils import mkdir_p
from torch_scatter import scatter_max
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from scene.basic_model import BasicModel
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, knn
    
class GaussianModel(BasicModel):
    def __init__(self, **model_kwargs):

        super().__init__(**model_kwargs)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.n_offsets = 1

    def eval(self):
        return

    def train(self):
        return
    
    def capture(self):
        return super().capture().update({
            "anchor": self._anchor,
            "offset": self._offset,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest
        })

    def restore(self, model_args, training_args):
        super().restore(model_args, training_args)
        self._anchor = model_args["anchor"]
        self._offset = model_args["offset"]
        self._scaling = model_args["scaling"]
        self._rotation = model_args["rotation"]
        self._opacity = model_args["opacity"]
        self._features_dc = model_args["features_dc"]
        self._features_rest = model_args["features_rest"]

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_offset(self):
        return self._offset
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def set_appearance(self, num_cameras):
        self.embedding_appearance = None

    def training_setup(self, training_args):
        super().training_setup(training_args)

        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "features_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "features_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
        if self.max_sh_degree is not None:
            self.update_sh_degree(iteration)

    def create_from_pcd(self, pcd, spatial_lr_scale, logger):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(pcd.points).float().cuda()
        colors = torch.tensor(pcd.colors).float().cuda()
        if self.voxel_size <= 0:
            init_dist = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            torch.cuda.empty_cache()
                
        fused_point_cloud, fused_color = self.voxelize_sample(points, colors)
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        logger.info(f'Initial Voxel Number: {fused_point_cloud.shape[0]}')
        logger.info(f'Voxel Size: {self.voxel_size}')

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

    def save_anchor_ply(self, path):
        def construct_list_of_attributes():
            l = ['x', 'y', 'z']
            for i in range(self._offset.shape[1]*self._offset.shape[2]):
                l.append('f_offset_{}'.format(i))
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offset, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_anchor_ply(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        features_dc = np.zeros((anchor.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((anchor.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_gaussian_ply(self, path):
        def construct_list_of_attributes():
            l = ['x', 'y', 'z']
            
            # All channels except the 3 DC
            for i in range(3):
                l.append('f_dc_{}'.format(i))
            for i in range(3 * (self.max_sh_degree + 1) ** 2 - 3):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(3):
                l.append('scale_{}'.format(i))
            for i in range(4):
                l.append('rot_{}'.format(i))
            return l

        xyz = self._anchor + self._offset.view([-1, 3]) * self._scaling[:,:3]
        xyz = xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling[:,3:].detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def add_anchor(self, candidate_anchor, cur_size, candidate_mask, remove_duplicates, inverse_indices):
        new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda() * cur_size # *0.05
        new_scaling = torch.log(new_scaling)
        new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
        new_rotation[:,0] = 1.0
        new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
        new_features = self.get_features.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, (self.max_sh_degree + 1) ** 2, 3])[:len(candidate_mask), :][candidate_mask]
        new_features = scatter_max(new_features, inverse_indices.unsqueeze(1).expand(-1, new_features.size(1)), dim=0)[0][remove_duplicates]
        new_features_dc = new_features[:, 0:1, :]
        new_features_rest = new_features[:, 1:, :]
        new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

        d = {
            "anchor": candidate_anchor,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "offset": new_offsets,
            "opacity": new_opacities,
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._anchor = optimizable_tensors["anchor"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._offset = optimizable_tensors["offset"]
        self._opacity = optimizable_tensors["opacity"]

    def generate_gaussians(self, viewpoint_camera, visible_mask=None):
        # view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        scaling = self.get_scaling[visible_mask]
        opacity = self.get_opacity[visible_mask]
        rotation = self.get_rotation[visible_mask]
        color = self.get_features[visible_mask]

        opacity = opacity * self.smooth_complement(visible_mask)

        # offsets
        offsets = grid_offsets.view([-1, 3]) * scaling[:,:3]
        scaling = scaling[:,3:] 
        
        xyz = anchor + offsets 
        mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")

        return xyz, color, opacity, scaling, rotation, self.active_sh_degree, mask