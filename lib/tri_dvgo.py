import os
import time
import functools
from matplotlib.pyplot import step
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo

from .backbone import make_edsr, make_edsr_baseline
from .mlp import Mapping, Interp_MLP
from .backbone import resnet_extractor
from .load_blender import pose_spherical

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)


'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=6, 
                 
                 rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 
                 interp_width=64,
                 interp_depth=2,
                 
                 tri_aggregation = 'concat',
                 feat_pe=0,
                 feat_fourier=False,

                 map_depth=5,
                 map_width=64,
                 liif=False,
                 no_voxel_feat=False,
                 posbase_pe=0, 
                 global_cell_decode=False,
                 implicit_voxel_feat=False, feat_unfold=False, local_ensemble=True, cell_decode=True,
                 cat_posemb=False,
                 name='edsr-baseline', n_feats=64, n_resblocks=16, res_scale=1, scale=2, no_upsampling=True, rgb_range=1,
                 pretrained_state_dict=None,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.liif = liif
        self.tri_aggregation = tri_aggregation
        self.rgbnet_dim = rgbnet_dim
        self.no_voxel_feat = no_voxel_feat
        self.cat_posemb = cat_posemb
        self.global_cell_decode = global_cell_decode
        self.feat_pe = feat_pe
        self.feat_fourier = feat_fourier
        if name == 'edsr-baseline':
            self.encoder_kwargs = {
                'n_resblocks': n_resblocks, 'n_feats': n_feats, 'res_scale': res_scale,  
                'scale': scale, 'no_upsampling': no_upsampling, 'rgb_range': rgb_range,
            }
            self.encoder = make_edsr_baseline(**self.encoder_kwargs)
            print('initialized encoder networks')
            print(self.encoder)
            if pretrained_state_dict:
                sd = torch.load(pretrained_state_dict)
                self.encoder.load_state_dict(sd)
                print('loaded edsr weights from: ', pretrained_state_dict)
            
        elif name == 'resnet34':
            self.encoder = resnet_extractor()
            self.encoder_kwargs = {
                'name': 'resnet34'
            }
            print('initialized encoder networks')
            print(self.encoder)
        
        else:
            raise NotImplementedError

        self.map = Mapping(in_dim=n_feats+16, out_dim=rgbnet_dim, depth=map_depth, width=map_width)
        print('initialized mapping networks')
        print(self.map)
        
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.posbase_pe = posbase_pe

        # set anchor poses
        pose_anchor = [
            pose_spherical(theta=0, phi=90, radius=4),  # xy
            pose_spherical(theta=90, phi=0, radius=4),  # yz
            pose_spherical(theta=90, phi=90, radius=4), # zx
        ]
        pose_anchor = torch.stack(pose_anchor, dim=0)
        assert pose_anchor.shape[0] == 3
        self.pose_anchor = pose_anchor.to(self.xyz_min.device)

        # liif configs
        self.implicit_voxel_feat = implicit_voxel_feat
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
            'posbase_pe': posbase_pe,
            'interp_width': interp_width,
            'interp_depth': interp_depth,
            'map_depth': map_depth,
            'map_width': map_width,
            'feat_pe': feat_pe,
            'feat_fourier': feat_fourier,
        }

        if implicit_voxel_feat:
            print("\n\033[96mimplicit voxel feat!!!\033[0m")
            if self.liif:
                dim0 = 2
            else:
                dim0 = (2+2*posbase_pe*2)
            if feat_unfold:
                dim0 += rgbnet_dim * 9
            else:
                dim0 += rgbnet_dim
            if cell_decode:
                dim0 += 2
            # self.interp = nn.Linear(dim0, rgbnet_dim)
            self.interp_xy = Interp_MLP(dim0, rgbnet_dim, width=interp_width, depth=interp_depth)
            self.interp_yz = Interp_MLP(dim0, rgbnet_dim, width=interp_width, depth=interp_depth)
            self.interp_zx = Interp_MLP(dim0, rgbnet_dim, width=interp_width, depth=interp_depth)
            self.interp = {
                'xy': self.interp_xy,
                'yz': self.interp_yz,
                'zx': self.interp_yz,
            }
            # self.interp = nn.Sequential(
            #     *[nn.Linear(dim0, interp_width), nn.ReLU(inplace=True)],
            #     *[
            #         nn.Sequential(nn.Linear(interp_width, interp_width), nn.ReLU(inplace=True))
            #         for _ in range(interp_depth-2)
            #     ],
            #     nn.Linear(interp_width, rgbnet_dim),
            # )
            print(self.interp)
            print('dvgo: dim0              ', dim0)
            print('dvgo: feat_unfold       ', self.feat_unfold)
            print('dvgo: cell_decode       ', self.cell_decode)
        
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            raise NotImplementedError
            # color voxel grid  (coarse stage)
            # self.k0_dim = 3
            # self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            # self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                if tri_aggregation == 'concat':
                    self.k0_dim = rgbnet_dim * 3 # tri-plane concat
                elif tri_aggregation == 'sum':
                    self.k0_dim = rgbnet_dim
            print(self.k0_dim)
            # self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            if posbase_pe > 0:
                self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            if feat_pe > 0:
                self.register_buffer('featfreq', torch.FloatTensor([(2**i) for i in range(feat_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            if posbase_pe > 0 and (self.cat_posemb or self.no_voxel_feat):
                dim0 += (3+3*posbase_pe*2)
            if rgbnet_direct and not self.no_voxel_feat:
                if feat_fourier:
                    dim0 += self.k0_dim+self.k0_dim*feat_pe*2
                else:
                    dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            if global_cell_decode:
                dim0 += 3
            
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            # print('dvgo: feature voxel grid', self.k0.shape)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(self.world_size), dtype=torch.bool)
        self.mask_cache = MaskCache(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('\ndvgo: voxel_size      ', self.num_voxels)
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            'implicit_voxel_feat': self.implicit_voxel_feat, 
            'feat_unfold': self.feat_unfold, 
            'local_ensemble': self.local_ensemble, 
            'cell_decode': self.cell_decode,
            'no_voxel_feat': self.no_voxel_feat,
            'cat_posemb': self.cat_posemb,
            'global_cell_decode': self.global_cell_decode,
            'liif': self.liif,
            'tri_aggregation': self.tri_aggregation,
            **self.rgbnet_kwargs,
            **self.encoder_kwargs,
        }

    def unfold_feat(self, inp):
        _, _, d, h, w = inp.shape

        inp = F.pad(inp, pad=(1,1,1,1,1,1), mode='replicate')

        lst = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    lst.append(inp[:, :, i:i+d, j:j+h, k:k+w])

        out = torch.cat(lst, dim=1)
        return out
    

    def make_coord(self, axis='xyz'):
        assert axis in ['xyz', 'xy', 'yz', 'zx']

        if axis == 'xyz':
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[-3]),
                    torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[-2]),
                    torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[-1]),
                ), 
            dim=-1)

            self_grid_xyz = ((self_grid_xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
            self_grid_xyz = self_grid_xyz.to(self.xyz_min.device).permute(3, 0, 1, 2).unsqueeze(0)
            return self_grid_xyz
        elif axis == 'xy':
            self_grid_xy = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[-3]),
                    torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[-2]),
                ), 
            dim=-1)
            self_grid_xy = ((self_grid_xy - self.xyz_min[[0, 1]]) / (self.xyz_max[[0, 1]] - self.xyz_min[[0, 1]])).flip((-1,)) * 2 - 1
            self_grid_xy = self_grid_xy.to(self.xyz_min.device).permute(2, 0, 1).unsqueeze(0)
            return self_grid_xy
        elif axis == 'yz':
            self_grid_yz = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[-2]),
                    torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[-1]),
                ), 
            dim=-1)

            self_grid_yz = ((self_grid_yz - self.xyz_min[[1, 2]]) / (self.xyz_max[[1, 2]] - self.xyz_min[[1, 2]])).flip((-1,)) * 2 - 1
            self_grid_yz = self_grid_yz
            return self_grid_yz.to(self.xyz_min.device).permute(2, 0, 1).unsqueeze(0)
        elif axis == 'zx':
            self_grid_zx = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[-1]),
                    torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[-3]),
                ), 
            dim=-1)

            self_grid_zx = ((self_grid_zx - self.xyz_min[[2, 0]]) / (self.xyz_max[[2, 0]] - self.xyz_min[[2, 0]])).flip((-1,)) * 2 - 1
            self_grid_zx = self_grid_zx.to(self.xyz_min.device).permute(2, 0, 1).unsqueeze(0)
            return self_grid_zx
    
    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        # if self.k0_dim > 0:
        #     self.k0 = torch.nn.Parameter(
        #         F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        # else:
        #     self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        
        if self.mask_cache_path: 
            # loaded from the coarse cache
            mask_cache = MaskCache(
                path=self.mask_cache_path,
                mask_cache_thres=self.mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
            ), -1)

            coarse_mask = mask_cache(self_grid_xyz)
        
        self_alpha = F.max_pool3d(self.activate_density(self.density), kernel_size=3, padding=1, stride=1)[0,0]
        
        mask = coarse_mask & (self_alpha>self.fast_color_thres) if self.mask_cache_path else (self_alpha>self.fast_color_thres)
        self.mask_cache = MaskCache(
            path=None, mask=mask,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.density, self.density.grad, weight, weight, weight, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.k0, self.k0.grad, weight, weight, weight, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def grid_sampler2D(self, xyz, grids, mode=None, align_corners=True): # , is_k0=False, stepsize=None)
        '''Wrapper for the 2D interp operation'''
        N, _ = xyz.shape
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        mode='bilinear'
        xy_feat = F.grid_sample(grids['xy'], ind_norm[...,[0, 1]], mode=mode, align_corners=align_corners)[0, :, 0, :].T
        yz_feat = F.grid_sample(grids['yz'], ind_norm[...,[1, 2]], mode=mode, align_corners=align_corners)[0, :, 0, :].T
        zx_feat = F.grid_sample(grids['zx'], ind_norm[...,[2, 0]], mode=mode, align_corners=align_corners)[0, :, 0, :].T
        
        if self.tri_aggregation == 'concat':
            feat = torch.cat([xy_feat, yz_feat, zx_feat], dim=-1)
        elif self.tri_aggregation == 'sum':
            feat = xy_feat + yz_feat + zx_feat

        if self.global_cell_decode:
            cell = torch.zeros(N, 3).to(self.xyz_min.device)
            cell[..., 0] = 1. / self.world_size[-3]
            cell[..., 1] = 1. / self.world_size[-2]
            cell[..., 2] = 1. / self.world_size[-1]

            feat = torch.cat([feat, cell], dim=-1)

        return feat
    
    def liif_interpolate(self, xyz, feats):
        # grids ['xy'] ['yz'] ['zx'] [1, c, h, w]
        N, _ = xyz.shape
        xyz = xyz.reshape(1,-1,3)
        coord = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        
        if self.cell_decode or self.global_cell_decode:
            cell = torch.zeros_like(xyz).to(self.xyz_min.device)
            cell[..., 0] = 1. / self.world_size[-3]
            cell[..., 1] = 1. / self.world_size[-2]
            cell[..., 2] = 1. / self.world_size[-1]
        
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        # low level of detail rx ry rz
        rx = 2 / self.world_size[-3] / 2
        ry = 2 / self.world_size[-2] / 2
        rz = 2 / self.world_size[-1] / 2
        r = torch.Tensor([rx, ry, rz])

        splits = {'xy': [0, 1], 'yz': [1, 2], 'zx': [2, 0]}
        interp_feats = []
        for s, idxs in splits.items():
            feat_coord = self.make_coord(axis=s)
            rx, ry = r[idxs]
            preds = []
            areas = []
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord[..., idxs].clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    q_feat = F.grid_sample(
                        feats[s], coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    
                    rel_coord = coord[..., idxs] - q_coord
                    rel_coord[:, :, 0] *= feats[s].shape[-2]
                    rel_coord[:, :, 1] *= feats[s].shape[-1]

                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                    if self.cell_decode:
                        rel_cell = cell[..., idxs].clone()
                        rel_cell[:, :, 0] *= feats[s].shape[-2]
                        rel_cell[:, :, 1] *= feats[s].shape[-1]
                        inp = torch.cat([inp, rel_cell], dim=-1)
                    
                    pred = self.interp[s](inp.squeeze(0))
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                    areas.append(area + 1e-9)
            
            tot_area = torch.stack(areas).sum(dim=0)
            if self.local_ensemble:
                t = areas[0]; areas[0] = areas[3]; areas[3] = t
                t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            for pred, area in zip(preds, areas):
                ret = ret + pred * (area / tot_area).unsqueeze(-1)
            
            interp_feats.append(ret.squeeze(0))

        if self.tri_aggregation == 'concat':
            interp_feats = torch.cat(interp_feats, dim=-1)
        elif self.tri_aggregation == 'sum':
            interp_feats = interp_feats[0] + interp_feats[1] + interp_feats[2]
        if self.global_cell_decode:
            interp_feats = torch.cat([interp_feats, cell.squeeze(0)], dim=-1)

        return interp_feats

    
    def interpolate(self, xyz, grids,mode=None, align_corners=True):
        N, _ = xyz.shape
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        mode='bilinear'
        bi_feats = {}
        bi_feats['xy'] = F.grid_sample(grids['xy'], ind_norm[...,[0, 1]], mode=mode, align_corners=align_corners)[0, :, 0, :].T
        bi_feats['yz'] = F.grid_sample(grids['yz'], ind_norm[...,[1, 2]], mode=mode, align_corners=align_corners)[0, :, 0, :].T
        bi_feats['zx'] = F.grid_sample(grids['zx'], ind_norm[...,[2, 0]], mode=mode, align_corners=align_corners)[0, :, 0, :].T
        
        if self.cell_decode or self.global_cell_decode:
            cell = torch.zeros(N, 3).to(self.xyz_min.device)
            cell[..., 0] = 1. / self.world_size[-3]
            cell[..., 1] = 1. / self.world_size[-2]
            cell[..., 2] = 1. / self.world_size[-1]

        splits = {'xy': [0, 1], 'yz': [1, 2], 'zx': [2, 0]}

        interp_feats = []
        for s, idxs in splits.items():
            grid_coord = self.make_coord(axis=s)
            q_coord = F.grid_sample(grid_coord, ind_norm[...,idxs], mode=mode, align_corners=align_corners)[0, :, 0, :].T
            pos_emb = (q_coord.unsqueeze(-1) * self.posfreq).flatten(-2)
            pos_emb = torch.cat([q_coord, pos_emb.sin(), pos_emb.cos()], -1)
            if self.cell_decode:
                feat = torch.cat([bi_feats[s], pos_emb, cell[..., idxs]], dim=-1)
            else:
                feat = torch.cat([bi_feats[s], pos_emb], dim=-1)
            
            interp_feat = self.interp[s](feat)
            interp_feats.append(interp_feat)
        
        if self.tri_aggregation == 'concat':
            interp_feats = torch.cat(interp_feats, dim=-1)
        elif self.tri_aggregation == 'sum':
            interp_feats = interp_feats[0] + interp_feats[1] + interp_feats[2]
        if self.global_cell_decode:
            interp_feats = torch.cat([interp_feats, cell], dim=-1)

        return interp_feats

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True, is_k0=False, stepsize=None):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if not self.implicit_voxel_feat or not is_k0:
            mode = 'bilinear'
            ret_lst = [
                # TODO: use `rearrange' to make it readable
                F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
                for grid in grids
            ]
            for i in range(len(grids)):
                if ret_lst[i].shape[-1] == 1:
                    ret_lst[i] = ret_lst[i].squeeze(-1)
            if len(ret_lst) == 1:
                return ret_lst[0]
        else:
            raise NotImplementedError

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=0, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        # print(rays_o.shape, rays_d.shape, self.xyz_min, self.xyz_max, near, far, stepdist)
        
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id
    
    def encode_feat(self, rgb_lr, pose_lr,):
        xyz_feats = self.encoder(rgb_lr)
        feats = {'xy': xyz_feats[[0]], 'yz': xyz_feats[[1]], 'zx': xyz_feats[[2]]}
        feats = {
            'xy': self.map(feats['xy'], pose_lr[0] - self.pose_anchor[0]), 
            'yz': self.map(feats['yz'], pose_lr[1] - self.pose_anchor[1]), 
            'zx': self.map(feats['zx'], pose_lr[2] - self.pose_anchor[2])
        }
        return feats

    def forward(self, rgb_lr, pose_lr, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        feats = self.encode_feat(rgb_lr, pose_lr)
        # self.k0 = feats
        ret_dict = self.render(feats, rays_o, rays_d, viewdirs, global_step, **render_kwargs)
        return ret_dict
    
    def render(self, feats, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        
        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = self.grid_sampler(ray_pts, self.density)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        
        # query for color
        if not self.rgbnet_full_implicit:
            # lego coarse self.k0 [1, 3, 107, 107, 88]
            #             k0 feature 12
            
            if self.implicit_voxel_feat:
                # raise NotImplementedError
                if self.feat_unfold:
                    _, c, h, w = feats['xy'].shape
                    for s in ['xy', 'yz', 'zx']:
                        feats[s] = F.unfold(feats[s], 3, padding=1).view(_, c * 9, h, w)
                if self.liif:
                    k0 = self.liif_interpolate(ray_pts, feats)
                else:
                    k0 = self.interpolate(ray_pts, feats)
            else:
                # bilinear sampler
                k0 = self.grid_sampler2D(ray_pts, feats)
        
        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]

            if self.feat_pe > 0 and self.feat_fourier:
                feat_emb = (k0_view.unsqueeze(-1) * self.featfreq).flatten(-2)
                k0_view = torch.cat([k0_view, feat_emb.sin(), feat_emb.cos()], -1)

            if self.posbase_pe > 0 and self.no_voxel_feat:
                # ray_pts already masked !
                pos_emb = (ray_pts.unsqueeze(-1) * self.posfreq).flatten(-2)
                pos_emb = torch.cat([ray_pts, pos_emb.sin(), pos_emb.cos()], -1)
                rgb_feat = torch.cat([pos_emb, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)
                rgb = torch.sigmoid(rgb_logit)
            else:
                if self.posbase_pe > 0 and self.cat_posemb:
                    pos_emb = (ray_pts.unsqueeze(-1) * self.posfreq).flatten(-2)
                    pos_emb = torch.cat([ray_pts, pos_emb.sin(), pos_emb.cos()], -1)
                    rgb_feat = torch.cat([k0_view, pos_emb, viewdirs_emb], -1)
                else:
                    rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)

                if self.rgbnet_direct:
                    rgb = torch.sigmoid(rgb_logit)
                else:
                    rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching weights [X, ] rgb [X, 3] rgb_marched [N_rays, 3]
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),  # step related to world size
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super().__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_kwargs']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval);
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None
