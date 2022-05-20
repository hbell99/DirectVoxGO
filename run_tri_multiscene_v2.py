import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import cv2
import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from lib import utils, dvgo_multiscene, tri_dvgo_multiscene, dmpigo, ray_utils
from lib.load_data import load_everything
from lib.load_blender import dataset_dict



def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--export_fine_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=50000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, lr_imgs=None, lr_poses=None, fixed_lr_imgs=None, fixed_lr_poses=None, savedir=None, render_factor=0,
                      scene_id=None,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if fixed_lr_imgs:
        rgb_lr = torch.stack(fixed_lr_imgs, dim=0)
        pose_lr = fixed_lr_poses
    else:
        assert len(lr_imgs) == len(lr_poses)
        lr_imgs = torch.stack(lr_imgs, dim=0)
        j = torch.randint(lr_poses.shape[0], [3])
        rgb_lr = lr_imgs[j]
        pose_lr = lr_poses[j]
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = ray_utils.get_training_rays(
                rgb_tr=rgb_lr,
                train_poses=pose_lr,
                HW=HW[:3], Ks=Ks[:3], ndc=ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    rgb_lr = torch.cat([rgb_lr, rays_o_tr, rays_d_tr], dim=-1) # , viewdirs_tr
    rgb_lr = rgb_lr.permute(0, 3, 1, 2)
    h, w = rgb_lr.shape[-2:]
    h, w = h // render_kwargs['render_down'], w // render_kwargs['render_down']
    resize = transforms.Resize([h, w])
    rgb_lr = resize(rgb_lr)
    rgb_lr = (rgb_lr - 0.5) / 0.5
    rgb_lr = rgb_lr.to(device)
    pose_lr = pose_lr.to(device)
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    print('rbg_lr shape:', rgb_lr.shape)
    # assert rgb_lr.shape[-1] == 200
    # _, feats, _, _ = model.encode_feat(rgb_lr, pose_lr, scene_id)
    feats = model.encode_feat_inference(rgb_lr, pose_lr, scene_id)
    print(feats['xy'].shape)
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = ray_utils.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        # rgb_lr = lr_imgs[i]
        # rgb_lr = rgb_lr.permute(0, 3, 1, 2).to(rays_o.device)
        # rgb_lr = transforms.ToTensor()(rgb_lr).unsqueeze(0).to(rays_o.device)
        # rgb_lr = (rgb_lr - 0.5) / 0.5
        
        render_result_chunks = [
            # {k: v for k, v in model(rgb_lr, pose_lr, ro.to(device), rd.to(device), vd.to(device), scene_id=scene_id, **render_kwargs)[0].items() if k in keys}
            {k: v for k, v in model.render(feats, ro.to(device), rd.to(device), vd.to(device), scene_id=scene_id, res=None, **render_kwargs)[0].items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)

    return rgbs, depths


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    # xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_min = torch.cuda.FloatTensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for scene_id in range(len(HW)):
        for (H, W), K, c2w in zip(HW[scene_id], Ks[scene_id], poses[scene_id]):
            # c2w = torch.Tensor(c2w)
            c2w = c2w.to(device)
            rays_o, rays_d, viewdirs = ray_utils.get_rays_of_a_view(
                    H=H, W=W, K=K, c2w=c2w,
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
            if cfg.data.ndc:
                pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
            else:
                pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
            xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
            xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres, n_scene):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    xyz_min_list = []
    xyz_max_list = []
    for scene_id in range(n_scene):
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, model.density[scene_id].shape[-3]),
            torch.linspace(0, 1, model.density[scene_id].shape[-2]),
            torch.linspace(0, 1, model.density[scene_id].shape[-1]),
        ), -1)
        interp = interp.to(model.xyz_min.device)
        dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
        density = model.grid_sampler(dense_xyz, model.density[scene_id])
        alpha = model.activate_density(density)
        mask = (alpha > thres)
        active_xyz = dense_xyz[mask]
        xyz_min = active_xyz.amin(0)
        xyz_max = active_xyz.amax(0)
        xyz_min_list.append(xyz_min)
        xyz_max_list.append(xyz_max)
    xyz_min = torch.stack(xyz_min_list, 0)
    xyz_max = torch.stack(xyz_max_list, 0)
    xyz_min = torch.min(xyz_min, dim=0).values
    xyz_max = torch.max(xyz_max, dim=0).values
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, multiscene_dataset, stage, coarse_ckpt_path=None):
    if stage == 'fine' and not cfg.fine_model_and_render.use_coarse_geo:
        coarse_ckpt_path = None
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        ckpts = [os.path.join(cfg.basedir, cfg.expname, f) for f in sorted(os.listdir(os.path.join(cfg.basedir, cfg.expname))) if 'tar' in f and 'fine' in f]
        if len(ckpts) > 0:
            reload_ckpt_path = ckpts[-1]
        else:
            reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
        # init model
        model_kwargs = copy.deepcopy(cfg_model)
        if cfg.data.ndc:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
            num_voxels = model_kwargs.pop('num_voxels')
            if len(cfg_train.pg_scale) and reload_ckpt_path is None:
                num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
            model = dmpigo.DirectMPIGO(
                xyz_min=xyz_min, xyz_max=xyz_max,
                num_voxels=num_voxels,
                mask_cache_path=coarse_ckpt_path,
                **model_kwargs)
        else:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
            num_voxels = model_kwargs.pop('num_voxels')
            if len(cfg_train.pg_scale) and reload_ckpt_path is None:
                num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
            if stage == 'coarse':
                model = dvgo_multiscene.DirectVoxGO(
                    xyz_min=xyz_min, xyz_max=xyz_max,
                    num_voxels=num_voxels,
                    mask_cache_path=coarse_ckpt_path,
                    **model_kwargs)
            else:
                m_path = None if cfg_model.n_scene == 8 else coarse_ckpt_path
                model = tri_dvgo_multiscene.DirectVoxGO(
                    xyz_min=xyz_min, xyz_max=xyz_max,
                    num_voxels=num_voxels,
                    mask_cache_path=m_path, #coarse_ckpt_path,
                    **model_kwargs)
            if cfg_model.maskout_near_cam_vox:
                model.maskout_near_cam_vox(multiscene_dataset.all_poses[:, :, :3, 3], multiscene_dataset.near)
        model = model.to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    else:
        print(f'\n\nscene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        else:
            if stage == 'fine':
                model_class = tri_dvgo_multiscene.DirectVoxGO
            else:
                model_class = dvgo_multiscene.DirectVoxGO
        model = utils.load_model(model_class, reload_ckpt_path).to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    # init rendering setup
    render_kwargs = {
        'near': multiscene_dataset.near,
        'far': multiscene_dataset.far,
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays(rgb_tr_ori, poses, HW, Ks, scene_id): 
        # single data !!! not for a batch data

        # rgb_tr_ori = rgb_tr_ori.to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = ray_utils.get_training_rays_in_maskcache_sampling_for_multiscene(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses,
                HW=HW, Ks=Ks,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, scene_id= scene_id, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'random':

            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = ray_utils.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses,
                HW=HW, Ks=Ks, ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            raise NotImplementedError
        
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, 
    
    # initialize dataloader
    # data_loader = torch.utils.data.DataLoader(
    #     multiscene_dataset,
    #     batch_size=cfg.data.batch_size,
    #     shuffle=True,
    #     num_workers=2, # 8
    #     # pin_memory=True,
    #     # generator=torch.Generator(device=device)
    # )

    # get training rays for all scenes
    all_rgb_tr, all_rays_o_tr, all_rays_d_tr, all_viewdirs_tr, all_imsz = [], [], [], [], []
    if cfg_train.ray_sampler == 'in_maskcache':
        all_batch_index_sampler = []

    for scene_id in range(cfg_model.n_scene):
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = gather_training_rays(
            multiscene_dataset.all_imgs[scene_id], multiscene_dataset.all_poses[scene_id], multiscene_dataset.all_HW[scene_id], multiscene_dataset.all_Ks[scene_id], scene_id)
        all_rgb_tr.append(rgb_tr)
        all_rays_o_tr.append(rays_o_tr)
        all_rays_d_tr.append(rays_d_tr)
        all_viewdirs_tr.append(viewdirs_tr)
        if cfg_train.ray_sampler == 'in_maskcache':
            index_generator = ray_utils.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
            batch_index_sampler = lambda: next(index_generator)
            all_batch_index_sampler.append(batch_index_sampler)
        

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            for scene_id in range(len(model.density)):
                self_alpha = F.max_pool3d(model.activate_density(model.density[scene_id]), kernel_size=3, padding=1, stride=1)[0,0]
                model.mask_cache.mask[scene_id] &= (self_alpha > model.fast_color_thres)

        assert model.mask_cache is not None

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, tri_dvgo_multiscene.DirectVoxGO):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dvgo_multiscene.DirectVoxGO):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.density.data.sub_(1)

        # load data from data loader
        scene_id = np.random.randint(cfg_model.n_scene)
        if cfg_train.ray_sampler == 'in_maskcache':
            if stage == 'fine':
                if cfg_train.fixed_lr_idx:
                    j = cfg_train.fixed_lr_idx
                else:
                    j = torch.randint(multiscene_dataset.all_imgs[scene_id].shape[0], [3])
                rgb_lr = multiscene_dataset.all_imgs[scene_id][j]
                pose_lr = multiscene_dataset.all_poses[scene_id][j]

            sel_i = all_batch_index_sampler[scene_id]
            target = all_rgb_tr[scene_id][sel_i]
            rays_o = all_rays_o_tr[scene_id][sel_i]
            rays_d = all_rays_d_tr[scene_id][sel_i]
            viewdirs = all_viewdirs_tr[scene_id][sel_i]
        
        elif cfg_train.ray_sampler == 'random':
            if stage == 'fine':
                if cfg_train.fixed_lr_idx:
                    j = cfg_train.fixed_lr_idx
                else:
                    j = torch.randint(multiscene_dataset.all_imgs[scene_id].shape[0], [3])
                rgb_lr = multiscene_dataset.all_imgs[scene_id][j].to(device)
                pose_lr = multiscene_dataset.all_poses[scene_id][j].to(device)

                rays_o_lr = all_rays_o_tr[scene_id][j].to(device)
                rays_d_lr = all_rays_d_tr[scene_id][j].to(device)
                viewdirs_lr = all_viewdirs_tr[scene_id][j]
                rgb_lr = torch.cat([rgb_lr, rays_o_lr, rays_d_lr], dim=-1)

            i = torch.randint(all_rgb_tr[scene_id].shape[0], [cfg_train.N_rand])

            sel_r = torch.randint(all_rgb_tr[scene_id].shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(all_rgb_tr[scene_id].shape[2], [cfg_train.N_rand])
            target = all_rgb_tr[scene_id][i, sel_r, sel_c]
            rays_o = all_rays_o_tr[scene_id][i, sel_r, sel_c]
            rays_d = all_rays_d_tr[scene_id][i, sel_r, sel_c]
            viewdirs = all_viewdirs_tr[scene_id][i, sel_r, sel_c]
        
        target = target.to(device)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        viewdirs = viewdirs.to(device)

        # volume rendering
        if stage == 'coarse':
            # raise NotImplementedError
            render_result = model(rays_o, rays_d, viewdirs, scene_id, res=[200, 200], global_step=global_step, **render_kwargs)
        else:
            rgb_lr = rgb_lr.permute(0, 3, 1, 2)
            assert rgb_lr.shape[1] == 9
            if cfg_train.dynamic_downsampling:
                down = torch.rand([1])[0] * (cfg_train.dynamic_down - 1) + 1
                h, w = rgb_lr.shape[-2:]
                h, w = int(h / down), int(w / down)
                resize = transforms.Resize([h, w])
                rgb_lr = resize(rgb_lr)
            rgb_lr = (rgb_lr - 0.5) / 0.5
            
            rgb_lr = rgb_lr.to(device)
            pose_lr = pose_lr.to(device)
            render_result, consistency_loss, cosine_loss, distillation_loss = model(rgb_lr, pose_lr, rays_o, rays_d, viewdirs, scene_id, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        rgb_loss = F.mse_loss(render_result['rgb_marched'], target)
        # rgb_loss = F.smooth_l1_loss(render_result['rgb_marched'], target)
        loss = cfg_train.weight_main * rgb_loss
        psnr = utils.mse2psnr(loss.detach())
        # psnr = utils.mse2psnr(F.mse_loss(render_result['rgb_marched'], target).detach())
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        
        if stage == 'fine':
            if cfg_train.weight_consistency > 0:
                loss += cfg_train.weight_consistency * consistency_loss
            if cfg_train.weight_cosine > 0:
                loss += cfg_train.weight_cosine * cosine_loss
            if cfg_train.weight_distillation > 0:
                loss += cfg_train.weight_distillation * distillation_loss
        else:
            cosine_loss = consistency_loss = distillation_loss = 0.
        loss.backward()

        if stage == 'fine':
            nn.utils.clip_grad_norm_(parameters=model.encoder.parameters(), max_norm=5)
            nn.utils.clip_grad_norm_(parameters=model.map.parameters(), max_norm=5)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor
            if i_opt_g == 0:
                _lr = param_group['lr']
        if not isinstance(distillation_loss, float):
            distillation_loss = distillation_loss.item()
        if not isinstance(consistency_loss, float):
            consistency_loss = consistency_loss.item()
        if not isinstance(cosine_loss, float):
            cosine_loss = cosine_loss.item()
        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.5f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'consistency: {consistency_loss:.5f} / '
                       f'cosine: {cosine_loss:.5f} / '
                       f'distillation: {distillation_loss:.5f} / '
                       f'lr: {_lr:.6f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, multiscene_dataset):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, HW=multiscene_dataset.all_HW, Ks=multiscene_dataset.all_Ks, poses=multiscene_dataset.all_poses, near=multiscene_dataset.near, far=multiscene_dataset.far)
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                multiscene_dataset=multiscene_dataset, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.data.ndc:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        if cfg.fine_model_and_render.use_coarse_geo:
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                    model_class=dvgo_multiscene.DirectVoxGO, model_path=coarse_ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres, n_scene=cfg.fine_model_and_render.n_scene)
        else:
            xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            multiscene_dataset=multiscene_dataset, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    # torch.multiprocessing.set_start_method('spawn')
    if torch.cuda.is_available():
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    # data_dict = load_everything(args=args, cfg=cfg)

    multiscene_dataset = dataset_dict[cfg.data.dataset](cfg.data.datadir, split='train', fixed_idx=cfg.fine_train.fixed_lr_idx, down=cfg.data.down, s=cfg.data.test_scenes)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = ray_utils.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo_multiscene.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    if args.export_fine_only:
        print('Export fine visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
            model = utils.load_model(tri_dvgo_multiscene.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_fine_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, multiscene_dataset)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        else:
            model_class = tri_dvgo_multiscene.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': multiscene_dataset.near,
                'far': multiscene_dataset.far,
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'render_down': cfg.data.render_down,
            },
        }
    scene2index = multiscene_dataset.scene2index
    print('scene2index', scene2index)

    del multiscene_dataset

    test_scenes = cfg.data.test_scenes
    cfg.data.down = 1
    basedir = cfg.data.datadir
    for s in test_scenes:
        print('testing scene', s)
        cfg.data.datadir = os.path.join(basedir, s)
        scene_id = scene2index[s]
        data_dict = load_everything(args=args, cfg=cfg)
        render_down = render_viewpoints_kwargs['render_kwargs']['render_down']
        # render trainset and eval
        if args.render_train:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}_testdown_{render_down}', s)
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    lr_imgs=[data_dict['images'][i] for i in data_dict['i_train']],
                    lr_poses=data_dict['poses'][data_dict['i_train']],
                    fixed_lr_imgs=[data_dict['images'][i] for i in data_dict['i_train'] if i in cfg.fine_train.fixed_lr_idx],
                    fixed_lr_poses=data_dict['poses'][data_dict['i_train']][cfg.fine_train.fixed_lr_idx],
                    savedir=testsavedir,
                    scene_id=scene_id,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

        # render testset and eval
        if args.render_test:
            render_down = render_viewpoints_kwargs['render_kwargs']['render_down']
            # assert render_down == 16
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}_testdown_{render_down}', s)
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_test']],
                    HW=data_dict['HW'][data_dict['i_test']],
                    Ks=data_dict['Ks'][data_dict['i_test']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                    lr_imgs=[data_dict['images'][i] for i in data_dict['i_train']],
                    lr_poses=data_dict['poses'][data_dict['i_train']],
                    fixed_lr_imgs=[data_dict['images'][i] for i in data_dict['i_train'] if i in cfg.fine_train.fixed_lr_idx_render],
                    fixed_lr_poses=data_dict['poses'][data_dict['i_train']][cfg.fine_train.fixed_lr_idx_render],
                    savedir=testsavedir,
                    scene_id=scene_id,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

        # render video
        if args.render_video:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, depths = render_viewpoints(
                    render_poses=data_dict['render_poses'],
                    HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    lr_imgs=[data_dict['images'][i] for i in data_dict['i_train']],
                    lr_poses=data_dict['poses'][data_dict['i_train']],
                    fixed_lr_imgs=[data_dict['images'][i] for i in data_dict['i_train'] if i in cfg.fine_train.fixed_lr_idx],
                    fixed_lr_poses=data_dict['poses'][data_dict['i_train']][cfg.fine_train.fixed_lr_idx],
                    savedir=testsavedir,
                    scene_id=scene_id,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    print('Done')

