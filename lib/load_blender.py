import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import pickle


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, down=0):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    if down > 1:
        H = H//down
        W = W//down
        focal = focal/float(down)

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    return imgs, poses, render_poses, [H, W, focal], i_split



def load_blender_data_lrsr(basedir, down=4, testskip=1):

    pkl_file = os.path.join(basedir, f'down_{down}.pkl')

    if os.path.isfile(pkl_file):
        with open(pkl_file, 'rb') as f:
            ret = pickle.load(f)
        return ret['imgs_lr'], ret['imgs_sr'], ret['poses'], ret['render_poses'], ret['sr_cam'], ret['lr_cam'], ret['i_split']
    
    
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs_sr = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs_sr = []
        imgs_lr = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs_sr.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs_sr = (np.array(imgs_sr) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs_sr.shape[0])
        all_imgs_sr.append(imgs_sr)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs_sr = np.concatenate(all_imgs_sr, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs_sr[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal_sr = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    

    h, w = H // down, W // down
    focal_lr = focal_sr / float(down)

    imgs_lr = np.zeros((imgs_sr.shape[0], h, w, 4))
    for i, img in enumerate(imgs_sr):
        imgs_lr[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    ret = {
        'imgs_lr': imgs_lr, 
        'imgs_sr': imgs_sr, 
        'poses': poses, 
        'render_poses': render_poses, 
        'sr_cam': [H, W, focal_sr], 
        'lr_cam': [h, w, focal_lr], 
        'i_split': i_split,
    }

    with open(pkl_file, 'wb')as f:
        pickle.dump(ret, f)
    
    return imgs_lr, imgs_sr, poses, render_poses, [H, W, focal_sr], [h, w, focal_lr], i_split