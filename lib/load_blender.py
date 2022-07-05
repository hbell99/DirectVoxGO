import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import pickle
from torch.utils.data import Dataset
from torchvision import transforms as T

from .load_nsvf import MultisceneNSVFDataset


# translation along x axis
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# rotation along x axis
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# rotation along y axis
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
    # imgs = torch.Tensor(imgs)
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

    if down > 1:
        imgs_lr = np.zeros((imgs_sr.shape[0], h, w, 4))
        for i, img in enumerate(imgs_sr):
            imgs_lr[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        imgs_lr = imgs_sr

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


class BlenderDataset(Dataset):
    H, W = 800, 800
    near, far = 2., 6.
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    def __init__(self, basedir, half_res=False, testskip=1, down=1, split='train', white_bkgd=True, fixed_idx=None):
        if split =='train' or testskip==0:
            self.skip = 1
        else:
            self.skip = testskip
        
        self.basedir = basedir
        self.white_bkgd = white_bkgd
        self.fixed_idx = fixed_idx

        self.read_meta(basedir, split)
        self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)

        self.K = np.array([
            [self.focal, 0, 0.5*self.W],
            [0, self.focal, 0.5*self.H],
            [0, 0, 1]
        ])

        poses = []
        for frame in self.meta['frames']:
            poses.append(np.array(frame['transform_matrix']))
        
        self.poses = np.stack(poses, 0)

    def read_meta(self, basedir, split='train'):
        with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
            meta = json.load(fp)
        
        self.meta = meta
        self.camera_angle_x = self.meta['camera_angle_x']

    
    def get_input_views(self, fixed_idx=None):
        if fixed_idx:
            assert len(fixed_idx) == 3
            idxs = fixed_idx
        else:
            idxs = np.random.permutation(len(self.meta['frames']))[:3]
        
        images, poses = [], []
        for index in idxs:
            image, pose = self.form_data(index)
            images.append(image)
            poses.append(pose)
        
        images = np.stack(images, 0)
        poses = np.stack(poses, 0)
        return images, poses
    
    def form_data(self, index):
        frame = self.meta['frames'][index]
        fname = os.path.join(self.basedir, frame['file_path'] + '.png')
        image = imageio.imread(fname)
        H, W = image.shape[:2]
        assert H == self.H
        assert W == self.W
        image = (np.array(image) / 255.).astype(np.float32)
        if self.white_bkgd:
            image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
        else:
            image = image[...,:3]*image[...,-1:]
        pose = self.poses[index]

        return image, pose

    def __len__(self):
        return len(self.meta['frames'])

    def __getitem__(self, index):
        image, pose = self.form_data(index)
        image, pose = image[None], pose[None] # [1, H, W, 3], # [1, 4, 4]

        input_images, imput_poses = self.get_input_views(fixed_idx=self.fixed_idx) # [3, H, W, 3], # [3, 4, 4]

        if len(self.K.shape) == 2:
            Ks = self.K[None].repeat(len(pose), axis=0)
        else:
            Ks = self.K
        
        HW = np.array([im.shape[:2] for im in image])

        return image, pose, HW, Ks, input_images, imput_poses


class MultisceneBlenderDataset(Dataset):
    H, W = 800, 800
    near, far = 2., 6.
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    def __init__(self, basedir, half_res=False, testskip=1, down=1, split='train', white_bkgd=True, fixed_idx=None):
        if split =='train' or testskip==0:
            self.skip = 1
        else:
            self.skip = testskip
        
        self.basedir = basedir
        self.white_bkgd = white_bkgd
        self.fixed_idx = fixed_idx

        self.read_meta(basedir, split)
        # self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)

        # self.K = np.array([
        #     [self.focal, 0, 0.5*self.W],
        #     [0, self.focal, 0.5*self.H],
        #     [0, 0, 1]
        # ])

        all_poses, all_Ks, all_HW = [], [], []
        for s in self.scenes:
            poses = []
            for frame in self.meta[s]['frames']:
                poses.append(np.array(frame['transform_matrix']))
            poses = np.array(poses).astype(np.float32)
            focal = .5 * self.W / np.tan(.5 * self.meta[s]['camera_angle_x'])
            K = np.array([
                [focal, 0, 0.5*self.W],
                [0, focal, 0.5*self.H],
                [0, 0, 1]
            ])
            if len(K.shape) == 2:
                Ks = K[None].repeat(len(poses), axis=0)
            else:
                Ks = K
            all_poses.append(poses)
            all_Ks.append(Ks)
            all_HW.append(np.array([[self.H, self.W] for i in range(len(self.meta[s]['frames']))]))
        
        self.all_poses = np.stack(all_poses, 0) # [n_scenes, n_views, 4, 4]
        self.all_Ks = np.stack(all_Ks, 0)
        self.all_HW = np.stack(all_HW, 0)

        # self.all_poses = torch.FloatTensor(self.all_poses)

    def read_meta(self, basedir, split='train'):
        metas = {}
        scenes = os.listdir(basedir)
        self.scenes = [s for s in scenes if not s.endswith('txt')]
        # self.scenes = ['lego']
        self.index2scene = {i: s for i, s in enumerate(self.scenes)}
        self.scene2index = {s: i for i, s in enumerate(self.scenes)}
        for s in self.scenes:
            with open(os.path.join(basedir, s, 'transforms_{}.json'.format(split)), 'r') as fp:
                metas[s] = json.load(fp)
        
        self.meta = metas

    
    def get_input_views(self, scene_index, fixed_idx=None):
        s = self.index2scene[scene_index]
        if fixed_idx:
            assert len(fixed_idx) == 3
            idxs = fixed_idx
        else:
            idxs = np.random.permutation(100)[:3] # np.random.permutation(len(self.meta[s]['frames']))[:3]
        
        images, poses = [], []
        for index in idxs:
            image, pose, _ = self.form_data(scene_index, index)
            images.append(image)
            poses.append(pose)
        
        images = np.stack(images, 0)
        poses = np.stack(poses, 0)
        return images, poses
    
    def form_data(self, scene_index, frame_index):
        s = self.index2scene[scene_index]
        
        frame = self.meta[s]['frames'][frame_index]
        fname = os.path.join(self.basedir, s, frame['file_path'] + '.png')
        image = imageio.imread(fname)
        H, W = image.shape[:2]
        assert H == self.H
        assert W == self.W
        image = (np.array(image) / 255.).astype(np.float32)
        if self.white_bkgd:
            image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
        else:
            image = image[...,:3]*image[...,-1:]
        pose = self.all_poses[scene_index][frame_index]

        K = self.all_Ks[scene_index][0]

        return image, pose, K

    def __len__(self):
        return len(self.meta) # n_scenes

    def __getitem__(self, index):
        # s = self.index2scene[index]
        random_index = np.random.randint(100) # len(self.meta[s]['frames']
        image, pose, K = self.form_data(scene_index=index, frame_index=random_index)
        image, pose = image[None], pose[None]
        # image, pose = [], []
        # for i in range(100):
        #     _image, _pose, K = self.form_data(scene_index=index, frame_index=i) # [1, H, W, 3], # [1, 4, 4]
        #     image.append(_image)
        #     pose.append(_pose)
        
        # image = np.stack(image, 0)
        # pose = np.stack(pose, 0)

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

class MultisceneBlenderDataset_v2(Dataset):
    H, W = 800, 800
    near, far = 2., 6.
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    def __init__(self, basedir, testskip=1, down=1, split='train', white_bkgd=True, fixed_idx=None, s=None):
        self.H = int(self.H // down)
        self.W = int(self.W // down)
        if split =='train' or testskip==0:
            self.skip = 1
        else:
            self.skip = testskip
        
        self.basedir = basedir
        self.white_bkgd = white_bkgd
        self.fixed_idx = fixed_idx
        self.s = s

        self.read_meta(basedir, split)
        # self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)

        # self.K = np.array([
        #     [self.focal, 0, 0.5*self.W],
        #     [0, self.focal, 0.5*self.H],
        #     [0, 0, 1]
        # ])

        all_imgs, all_poses, all_Ks, all_HW = [], [], [], []
        for s in self.scenes:
            poses = []
            imgs = []
            for frame in self.meta[s]['frames']:
                fname = os.path.join(basedir, s, frame['file_path'] + '.png')
                image = imageio.imread(fname)

                image = (np.array(image) / 255.).astype(np.float32)
                if down > 1:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                
                H, W = image.shape[:2]
                if self.white_bkgd:
                    image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
                else:
                    image = image[...,:3]*image[...,-1:]
                imgs.append(image)
                poses.append(np.array(frame['transform_matrix']))
            poses = np.array(poses).astype(np.float32)
            imgs = np.stack(imgs, 0)

            focal = .5 * self.W / np.tan(.5 * self.meta[s]['camera_angle_x'])
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
            all_HW.append(np.array([[self.H, self.W] for i in range(len(self.meta[s]['frames']))]))
        
        self.all_imgs = torch.Tensor(np.stack(all_imgs, 0))
        self.all_poses = torch.Tensor(np.stack(all_poses, 0)) # [n_scenes, n_views, 4, 4]
        self.all_Ks = np.stack(all_Ks, 0)
        self.all_HW = np.stack(all_HW, 0)


    def read_meta(self, basedir, split='train'):
        metas = {}
        scenes = os.listdir(basedir)
        if self.s is None:
            self.scenes = [s for s in scenes if not s.endswith('txt')]
        else:
            self.scenes = self.s
        self.index2scene = {i: s for i, s in enumerate(self.scenes)}
        self.scene2index = {s: i for i, s in enumerate(self.scenes)}
        print('scene2index', self.scene2index)
        for s in self.scenes:
            with open(os.path.join(basedir, s, 'transforms_{}.json'.format(split)), 'r') as fp:
                metas[s] = json.load(fp)
        
        self.meta = metas

    
    def get_input_views(self, scene_index, fixed_idx=None):
        s = self.index2scene[scene_index]
        if fixed_idx:
            assert len(fixed_idx) == 3
            idxs = fixed_idx
        else:
            idxs = np.random.permutation(100)[:3] # np.random.permutation(len(self.meta[s]['frames']))[:3]
        
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

dataset_dict = {
    'MultisceneBlenderDataset_v2': MultisceneBlenderDataset_v2, 
    'MultisceneBlenderDataset': MultisceneBlenderDataset,
    'MultisceneNSVFDataset': MultisceneNSVFDataset,
}