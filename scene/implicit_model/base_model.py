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
import time
import torch
import math
import numpy as np
from torch import nn
from einops import repeat
from functools import reduce
from torch_scatter import scatter_max
from utils.general_utils import get_expon_lr_func, knn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from scene.embedding import Embedding
from scene.basic_model import BasicModel
from utils.system_utils import searchForMaxIteration
    
class GaussianModel(BasicModel):
    def __init__(self, **model_kwargs):

        super().__init__(**model_kwargs)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.ape_code = -1

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.color_dim*self.n_offsets),
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    def capture(self):
        capture_dict = super().capture().update({
            'anchor': self._anchor,
            'offset': self._offset,
            'anchor_feat': self._anchor_feat,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'mlp_opacity_dict': self.mlp_opacity.state_dict(),
            'mlp_cov_dict': self.mlp_cov.state_dict(),
            'mlp_color_dict': self.mlp_color.state_dict(),
        })
        
        if self.appearance_dim > 0:
            capture_dict.update({
                'embedding_appearance_dict': self.embedding_appearance.state_dict()
            })
            
        return capture_dict

    def restore(self, model_args, training_args):
        super().restore(model_args, training_args)
        self._anchor = model_args['anchor']
        self._offset = model_args['offset']
        self._anchor_feat = model_args['anchor_feat']
        self._scaling = model_args['scaling']
        self._rotation = model_args['rotation']
        self.mlp_opacity.load_state_dict(model_args['mlp_opacity_dict'])
        self.mlp_cov.load_state_dict(model_args['mlp_cov_dict'])
        self.mlp_color.load_state_dict(model_args['mlp_color_dict'])
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(model_args['embedding_appearance_dict'])

    @property
    def get_anchor(self):
        return self._anchor
        
    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
            
    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
        else:
            self.embedding_appearance = None

    def training_setup(self, training_args):
        super().training_setup(training_args)

        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
        if self.max_sh_degree is not None:
            self.update_sh_degree(iteration)
    def create_from_pcd(self, pcd, spatial_lr_scale, logger):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(pcd.points).float().cuda()
        if self.voxel_size <= 0:
            init_dist = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            torch.cuda.empty_cache()
                        
        fused_point_cloud = self.voxelize_sample(points)
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        logger.info(f'Initial Voxel Number: {fused_point_cloud.shape[0]}')
        logger.info(f'Voxel Size: {self.voxel_size}')

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
                                  
    def save_anchor_ply(self, path):
        def construct_list_of_attributes():
            l = ['x', 'y', 'z']
            for i in range(self._offset.shape[1]*self._offset.shape[2]):
                l.append('f_offset_{}'.format(i))
            for i in range(self._anchor_feat.shape[1]):
                l.append('f_anchor_feat_{}'.format(i))
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offset, anchor_feat, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_anchor_ply(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
    
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
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

        anchor = self.get_anchor
        feat = self.get_anchor_feat
        grid_offsets = self.get_offset
        grid_scaling = self.get_scaling

        # get offset's opacity
        neural_opacity = self.get_opacity_mlp(feat) # [N, k]

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            camera_indicies = torch.zeros_like(feat[:,0], dtype=torch.long, device=feat.device)
            appearance = self.get_appearance(camera_indicies)
            color = self.get_color_mlp(torch.cat([feat, appearance], dim=1))
        else:
            color = self.get_color_mlp(feat)

        if self.color_attr == "RGB": 
            color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]
        else:
            color = color.reshape([anchor.shape[0]*self.n_offsets, -1])# [mask]
        color_dim = color.shape[1]

        # get offset's cov
        scale_rot = self.get_cov_mlp(feat)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, color_dim, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # post-process color
        color = color.view([color.shape[0], -1, 3])
        features_dc = color[:, 0:1, :]
        features_rest = color[:, 1:, :]

        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 
        
        xyz = xyz.detach().cpu().numpy()
        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale = scaling.detach().cpu().numpy()
        rotation = rot.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_mlp_checkpoints(self, path):
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.appearance_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        if self.appearance_dim > 0:
            emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_appearance.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()

    def prune_anchor(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def add_anchor(self, candidate_anchor, cur_size, candidate_mask, remove_duplicates, inverse_indices):
        new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
        new_scaling = torch.log(new_scaling)
        new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
        new_rotation[:,0] = 1.0
        new_feat = self.get_anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[:len(candidate_mask), :][candidate_mask]
        new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
        new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

        d = {
            "anchor": candidate_anchor,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "anchor_feat": new_feat,
            "offset": new_offsets,
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._anchor = optimizable_tensors["anchor"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._offset = optimizable_tensors["offset"]

    def generate_gaussians(self, viewpoint_camera, visible_mask=None):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        feat = self.get_anchor_feat[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]

        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        if self.view_dim > 0:
            cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, c+3]
        else:
            cat_local_view = feat # [N, c]

        if self.appearance_dim > 0:
            if self.ape_code < 0:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
                appearance = self.get_appearance(camera_indicies)
            else:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * self.ape_code
                appearance = self.get_appearance(camera_indicies)
                
        # get offset's opacity
        neural_opacity = self.get_opacity_mlp(cat_local_view) * self.smooth_complement(visible_mask)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)

        color = color.reshape([anchor.shape[0]*self.n_offsets, self.color_dim])# [mask]

        # get offset's cov
        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, self.color_dim, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 

        if self.color_attr != "RGB": 
            color = color.reshape([color.shape[0], self.color_dim // 3, 3])

        return xyz, color, opacity, scaling, rot, self.active_sh_degree, mask