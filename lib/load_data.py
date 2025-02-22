import numpy as np
import torch
import os

from .load_llff import load_llff_data
from .load_blender import load_blender_data, load_blender_data_lrsr
from .load_nsvf import load_nsvf_data
from .load_blendedmvs import load_blendedmvs_data
from .load_tankstemple import load_tankstemple_data
from .load_deepvoxels import load_dv_data
from .load_co3d import load_co3d_data


def load_data(args):

    K, depths = None, None

    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test = load_llff_data(
                args.datadir, args.factor, args.width, args.height,
                recenter=True, bd_factor=.75,
                spherify=args.spherify,
                load_depths=args.load_depths)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        if args.task == 'sr':
            images_lr, images, poses, render_poses, hwf, hwf_lr, i_split = load_blender_data_lrsr(basedir=args.datadir, down=args.down, testskip=args.testskip)
            print('Loaded sr blender', images.shape, images_lr.shape, render_poses.shape, hwf, hwf_lr, args.datadir)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, args.down)
            print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
                if args.task == 'sr':
                    images_lr = images_lr[...,:3]*images_lr[...,-1:] + (1.-images_lr[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]
                if args.task == 'sr':
                    images_lr = images_lr[...,:3]*images_lr[...,-1:]

    elif args.dataset_type == 'blendedmvs':
        images, poses, render_poses, hwf, K, i_split = load_blendedmvs_data(args.datadir)
        print('Loaded blendedmvs', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(args.datadir)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'nsvf':
        images, poses, render_poses, hwf, i_split = load_nsvf_data(args.datadir, args.down)
        print('Loaded nsvf', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.scene, basedir=args.datadir, testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R - 1
        far = hemi_R + 1
        assert args.white_bkgd
        assert images.shape[-1] == 3

    elif args.dataset_type == 'co3d':
        # each image can be in different shapes and intrinsics
        images, masks, poses, render_poses, hwf, K, i_split = load_co3d_data(args)
        print('Loaded co3d', args.datadir, args.annot_path, args.sequence_name)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        for i in range(len(images)):
            if args.white_bkgd:
                images[i] = images[i] * masks[i][...,None] + (1.-masks[i][...,None])
            else:
                images[i] = images[i] * masks[i][...,None]

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    if args.task == 'sr':
        HW_lr = np.array([im.shape[:2] for im in images_lr])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        if args.task == 'sr':
            H_lr, W_lr, focal_lr = hwf_lr
            H_lr, W_lr = int(H_lr), int(W_lr)
            hwf = [H_lr, W_lr, focal]
            K_lr = np.array([
                [focal_lr, 0, 0.5*W_lr],
                [0, focal_lr, 0.5*H_lr],
                [0, 0, 1]
            ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
        if args.task == 'sr':
            Ks_lr = K_lr[None].repeat(len(poses), axis=0)
    else:
        Ks = K
        if args.task == 'sr':
            Ks_lr = K_lr

    render_poses = render_poses[...,:4]

    if args.task == 'sr':
        data_dict = dict(
            hwf=hwf, hwf_lr=hwf_lr, HW=HW, HW_lr=HW_lr, Ks=Ks, Ks_lr=Ks_lr, near=near, far=far,
            i_train=i_train, i_val=i_val, i_test=i_test,
            poses=poses, render_poses=render_poses,
            images_lr=images_lr, 
            images=images, depths=depths,
            irregular_shape=irregular_shape,
        )

    else:
        data_dict = dict(
            hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
            i_train=i_train, i_val=i_val, i_test=i_test,
            poses=poses, render_poses=render_poses,
            images=images, depths=depths,
            irregular_shape=irregular_shape,
        )

    return data_dict


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    if cfg.data.task == 'sr':
        kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'images_lr', 'hwf_lr', 'HW_lr', 'Ks_lr'}
    else:
        kept_keys = {
                'hwf', 'HW', 'Ks', 'near', 'far',
                'i_train', 'i_val', 'i_test', 'irregular_shape',
                'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
        if cfg.data.task == 'sr':
            data_dict['images_lr'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images_lr']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
        if cfg.data.task == 'sr':
            data_dict['images_lr'] = torch.FloatTensor(data_dict['images_lr'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

