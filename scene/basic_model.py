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
from torch import nn
from functools import reduce
from einops import repeat
import numpy as np
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p
from torch_scatter import scatter_max
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid
    
class BasicModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self, **model_kwargs):
        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self.offset_opacity_accum = torch.empty(0)
        self.anchor_opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.padding = 0.0  
        self.setup_functions()

        if self.color_attr == "RGB":     
            self.active_sh_degree = None
            self.max_sh_degree = None   
            self.color_dim = 3
        else:
            self.active_sh_degree = 0
            self.max_sh_degree = int(''.join(filter(str.isdigit, self.color_attr)))
            self.color_dim = 3 * ((self.max_sh_degree + 1) ** 2)

    def capture(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "offset_opacity_accum": self.offset_opacity_accum,
            "anchor_opacity_accum": self.anchor_opacity_accum,
            "anchor_demon": self.anchor_demon,
            "offset_gradient_accum": self.offset_gradient_accum,
            "offset_denom": self.offset_denom,
            "max_radii2D": self.max_radii2D,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
        }

    def restore(self, model_args, training_args):

        self.active_sh_degree = model_args["active_sh_degree"]
        self.offset_opacity_accum = model_args["offset_opacity_accum"]
        self.anchor_opacity_accum = model_args["anchor_opacity_accum"]
        self.anchor_demon = model_args["anchor_demon"]
        self.offset_gradient_accum = model_args["offset_gradient_accum"]
        self.offset_denom = model_args["offset_denom"]
        self.max_radii2D = model_args["max_radii2D"]
        opt_dict = training_args["optimizer"].state_dict()
        self.spatial_lr_scale = training_args["spatial_lr_scale"]

        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        
    def voxelize_sample(self, points, colors=None):
        candidate_points = torch.round(points/self.voxel_size)
        candidates_unique, inverse_indices = torch.unique(candidate_points, return_inverse=True, dim=0)
        points = (candidates_unique + self.padding)* self.voxel_size  
        if colors is not None:
            colors = scatter_max(colors, inverse_indices.unsqueeze(1).expand(-1, points.size(1)), dim=0)[0]
            return points, RGB2SH(colors)
        else:
            return points

    def octree_sample(self, points, colors=None):
        positions = torch.empty(0, 3).float().cuda()
        levels = torch.empty(0).int().cuda() 
        if colors is not None:
            rgbs = torch.empty(0, 3).float().cuda()
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_candidates = torch.round((points - self.init_pos) / cur_size)
            new_candidates_unique, inverse_indices = torch.unique(new_candidates, return_inverse=True, dim=0)
            new_positions = new_candidates_unique * cur_size + self.init_pos
            new_positions += self.padding * cur_size
            new_levels = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            positions = torch.concat((positions, new_positions), dim=0)
            levels = torch.concat((levels, new_levels), dim=0)
            if colors is not None:
                new_rgbs = scatter_max(colors, inverse_indices.unsqueeze(1).expand(-1, colors.size(1)), dim=0)[0]
                rgbs = torch.concat((rgbs, new_rgbs), dim=0)
        if colors is not None:
            return positions, rgbs, levels
        else:
            return positions, levels

    def smooth_complement(self, visible_mask): 
        return torch.ones((visible_mask.sum(), 1), dtype=torch.float, device="cuda")

    def set_anchor_mask(self, *args):
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def training_setup(self, training_args):
        
        self.anchor_opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_opacity_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros(self.get_anchor.shape[0]*self.n_offsets, dtype=torch.float, device="cuda")
        
    def update_sh_degree(self, iteration):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.oneupSHdegree()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, render_pkg, growing_strategy, width, height):
        offset_selection_mask = render_pkg["selection_mask"]
        anchor_visible_mask = render_pkg["visible_mask"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        update_filter = render_pkg["visibility_filter"]
        opacity = render_pkg["opacity"]
        radii = render_pkg["radii"]
        
        # update opacity stats
        temp_opacity = torch.zeros(offset_selection_mask.shape[0], dtype=torch.float32, device="cuda")
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        temp_mask = offset_selection_mask.view([-1, self.n_offsets])
        sum_of_elements = temp_opacity.sum(dim=1, keepdim=True).cuda()
        count_of_elements = temp_mask.sum(dim=1, keepdim=True).float().cuda()
        average = sum_of_elements / torch.clamp(count_of_elements, min=1.0)
        average[count_of_elements == 0] = 0 # avoid nan
        self.anchor_opacity_accum[anchor_visible_mask] += average
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        grad_norm = torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        if growing_strategy=="mean":
            self.offset_gradient_accum[combined_mask] += grad_norm
        elif growing_strategy=="max":
            self.offset_gradient_accum[combined_mask] = torch.max(self.offset_gradient_accum[combined_mask], torch.abs(grad_norm))
            self.max_radii2D[combined_mask] = torch.max(self.max_radii2D[combined_mask], radii[update_filter])
            self.offset_opacity_accum[combined_mask] += opacity.clone().detach()[update_filter]
        else:
            raise ValueError(f"Unknown growing_type: {growing_strategy}")
        
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates

    def anchor_growing(self, grads, opt, offset_mask):
        grads[~offset_mask] = 0.0
        cur_size = self.voxel_size
        # update threshold
        cur_threshold = opt.densify_grad_threshold
        # mask from grad threshold
        candidate_mask = (grads >= cur_threshold)
        candidate_mask = torch.logical_and(candidate_mask, offset_mask)
        
        all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
        
        grid_coords = torch.round(self.get_anchor / cur_size - self.padding).int()
        selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
        selected_grid_coords = torch.round(selected_xyz / cur_size - self.padding).int()
        selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
        if opt.overlap:
            remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.padding * cur_size
        elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
            remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.padding * cur_size
        else:
            candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
            remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')

        if candidate_anchor.shape[0] > 0:

            self.add_anchor(candidate_anchor, cur_size, candidate_mask, remove_duplicates, inverse_indices)

            temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
            del self.anchor_demon
            self.anchor_demon = temp_anchor_demon

            temp_opacity_accum = torch.cat([self.anchor_opacity_accum, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
            del self.anchor_opacity_accum
            self.anchor_opacity_accum = temp_opacity_accum

            torch.cuda.empty_cache()

    def octree_growing(self, grads, opt, offset_mask, iteration):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** opt.update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = opt.densify_grad_threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * opt.extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += opt.extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if opt.overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                _, weed_mask = self.weed_out(candidate_anchor, new_level)
                candidate_anchor = candidate_anchor[weed_mask]
                new_level = new_level[weed_mask]
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                _, weed_mask = self.weed_out(candidate_anchor, new_level)
                candidate_anchor = candidate_anchor[weed_mask]
                new_level = new_level[weed_mask]
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size-self.padding).int()
            selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
            selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size-self.padding).int()
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                if opt.overlap:
                    remove_duplicates_ds =  torch.ones(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device="cuda")
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    candidate_anchor_ds = candidate_anchor_ds[weed_ds_mask]
                    new_level_ds = new_level_ds[weed_ds_mask]
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                elif selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    candidate_anchor_ds = candidate_anchor_ds[weed_ds_mask]
                    new_level_ds = new_level_ds[weed_ds_mask]
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                new_extra_level = torch.zeros(new_anchor.shape[0], dtype=torch.float, device='cuda')
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
                
                self.add_anchor(candidate_anchor, cur_size, candidate_mask, remove_duplicates, inverse_indices)
                self.add_anchor(candidate_anchor_ds, ds_size, candidate_ds_mask, remove_duplicates_ds, inverse_indices_ds)

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.anchor_opacity_accum, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_opacity_accum
                self.anchor_opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
    def run_densify(self, opt, iteration):
        # adding anchors
        if opt.growing_strategy=="mean":
            grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
            grads[grads.isnan()] = 0.0
            grads_norm = torch.norm(grads, dim=-1)
            offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        elif opt.growing_strategy=="max":
            grads = self.offset_gradient_accum # [N*k, 1]
            grads[grads.isnan()] = 0.0
            
            opacities = self.offset_opacity_accum/self.offset_denom
            opacities[opacities.isnan()] = 0.0
            opacities = opacities.flatten() # [N*k]

            grads_norm = torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(opacities, 1/5.0)
            offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
            offset_mask = torch.logical_and(offset_mask, opacities > 0.15)
        else:
            raise ValueError(f"Unknown growing_type: {opt.growing_strategy}")

        if self.__class__.__name__ == "GaussianModel":
            self.anchor_growing(grads_norm, opt, offset_mask)
        elif self.__class__.__name__ == "GaussianLoDModel": 
            self.octree_growing(grads_norm, opt, offset_mask, iteration)
        else:
            raise ValueError(f"Unknown model type: {self.gaussians.__class__.__name__}")
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        self.offset_opacity_accum[offset_mask] = 0
        padding_offset_opacity_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_opacity_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_opacity_accum.device)
        self.offset_opacity_accum = torch.cat([self.offset_opacity_accum, padding_offset_opacity_accum], dim=0)
        
        # prune anchors
        prune_mask = (self.anchor_opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        anchor_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchor_mask) # [N] 
        self.prune_anchor(prune_mask)
                
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        offset_opacity_accum = self.offset_opacity_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_opacity_accum = offset_opacity_accum.view([-1, 1])
        del self.offset_opacity_accum
        self.offset_opacity_accum = offset_opacity_accum
        
        # update opacity accum 
        if anchor_mask.sum()>0:
            self.anchor_opacity_accum[anchor_mask] = torch.zeros([anchor_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchor_mask] = torch.zeros([anchor_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.anchor_opacity_accum[~prune_mask]
        del self.anchor_opacity_accum
        self.anchor_opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon
        
        self.max_radii2D = torch.zeros(self.get_anchor.shape[0]*self.n_offsets, dtype=torch.float, device="cuda")

    def clean(self):
        del self.offset_opacity_accum
        del self.anchor_opacity_accum
        del self.anchor_demon
        del self.offset_gradient_accum
        del self.offset_denom
        del self.max_radii2D
        torch.cuda.empty_cache()