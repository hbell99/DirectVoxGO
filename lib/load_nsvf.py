import os
import glob
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

from torch.utils.data import Dataset
from torchvision import transforms as T

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

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


def load_nsvf_data(basedir, down=1):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))

    all_poses = []
    all_imgs = []
    i_split = [[], [], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)

    H, W = imgs[0].shape[:2]
    with open(os.path.join(basedir, 'intrinsics.txt')) as f:
        focal = float(f.readline().split()[0])

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if down > 1:
        H = H//down
        W = W//down
        focal = focal/float(down)

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    return imgs, poses, render_poses, [H, W, focal], i_split


class MultisceneNSVFDataset(Dataset):
    H, W = 800, 800
    # near, far = 2., 6.
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    splits = ['train', 'val', 'test']
    split2index = {s: i for i, s in enumerate(splits)}

    def __init__(self, basedir, down=1, split='train', white_bkgd=True, fixed_idx=None):
        self.H = self.H // down
        self.W = self.W // down
        
        self.basedir = basedir
        self.white_bkgd = white_bkgd
        self.fixed_idx = fixed_idx

        self.scenes = os.listdir(basedir)
        self.index2scene = {i: s for i, s in enumerate(self.scenes)}
        self.scene2index = {s: i for i, s in enumerate(self.scenes)}

        all_imgs, all_poses, all_Ks, all_HW = [], [], [], []
        for s in self.scenes:
            pose_paths = sorted(glob.glob(os.path.join(basedir, s, 'pose', '*txt')))
            rgb_paths = sorted(glob.glob(os.path.join(basedir, s, 'rgb', '*png')))
            
            poses = []
            imgs = []
            for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
                i_set = int(os.path.split(rgb_path)[-1][0])
                if i_set != self.split2index[split]:
                    continue
                
                image = imageio.imread(rgb_path)
                H, W = image.shape[:2]
                assert H == 800 and W == 800

                image = (np.array(image) / 255.).astype(np.float32)
                if down > 1:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                
                if image.shape[-1] == 4:
                    if white_bkgd:
                        image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
                    else:
                        image = image[...,:3]*image[...,-1:]
                
                imgs.append(image)
                poses.append(np.loadtxt(pose_path).astype(np.float32))
            poses = np.array(poses).astype(np.float32)
            imgs = np.stack(imgs, 0)

            with open(os.path.join(basedir, s, 'intrinsics.txt')) as f:
                focal = float(f.readline().split()[0])
            if down > 1:
                focal = focal / float(down)
            
            K = np.array([
                [focal, 0, 0.5*self.W],
                [0, focal, 0.5*self.H],
                [0, 0, 1]
            ])
            if len(K.shape) == 2:
                Ks = K[None].repeat(len(poses), axis=0)
            else:
                Ks = K
            
            all_imgs.append(imgs)
            all_poses.append(poses)
            all_Ks.append(Ks)
            all_HW.append(np.array([[self.H, self.W] for i in range(len(poses))]))
        
        self.all_imgs = torch.Tensor(np.stack(all_imgs, 0))
        self.all_poses = torch.Tensor(np.stack(all_poses, 0)) # [n_scenes, n_views, 4, 4]
        self.all_Ks = np.stack(all_Ks, 0)
        self.all_HW = np.stack(all_HW, 0)

        poses = self.all_poses.reshape(-1, 4, 4)
        self.near, self.far = inward_nearfar_heuristic(poses[:, :3, 3], ratio=0)

        print(self.all_imgs.shape)
        
    def get_input_views(self, scene_index, fixed_idx=None):
        s = self.index2scene[scene_index]
        if fixed_idx:
            assert len(fixed_idx) == 3
            idxs = fixed_idx
        else:
            idxs = np.random.permutation(len(self.all_poses[0]))[:3] # np.random.permutation(len(self.meta[s]['frames']))[:3]
        
        images = self.all_imgs[scene_index][idxs]
        poses = self.all_poses[scene_index][idxs]
        return images, poses
    
    def __len__(self):
        return len(self.meta) # n_scenes

    def __getitem__(self, index):
        # s = self.index2scene[index]
        # single image
        random_index = np.random.randint(100) # len(self.meta[s]['frames']
        image = self.all_imgs[index, random_index]
        pose = self.all_poses[index, random_index]
        K = self.all_Ks[index, random_index]
        image, pose = image[None], pose[None]

        # # all images
        # image = self.all_imgs[index]
        # pose = self.all_poses[index]
        # K = self.all_Ks[index]

        input_images, imput_poses = self.get_input_views(scene_index=index, fixed_idx=self.fixed_idx) # [3, H, W, 3], # [3, 4, 4]

        if len(K.shape) == 2:
            Ks = K[None].repeat(len(pose), axis=0)
        else:
            Ks = K
        
        HW = np.array([im.shape[:2] for im in image])

        # image = torch.FloatTensor(image, device='cpu')
        # poses = torch.FloatTensor(poses, device='cpu')
        # input_images = torch.FloatTensor(input_images, device='cpu')
        # input_poses = torch.FloatTensor(input_poses, device='cpu')

        scene_id = index

        return image, pose, HW, Ks, input_images, imput_poses, scene_id 