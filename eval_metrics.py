import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from nbformat import read
from tqdm import tqdm, trange

import cv2
import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils

def read_image(path):
    image = imageio.imread(path)
    image = (np.array(image) / 255.).astype(np.float32)
    if image.shape[-1] == 4:
        image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
    return image

if __name__ == '__main__':
    for s in ['Bike', 'Lifestyle', 'Palace', 'Robot', 'Spaceship', 'Steamtrain', 'Toad', 'Wineholder']:
        data_dir = f'data/Synthetic_NSVF/{s}'
        eval_dir = f'logs/nsvf_synthetic/dvgo_{s}/render_liif'
        psnrs = []
        ssims = []
        lpips_vgg = []
        rgb_paths = sorted(glob.glob(os.path.join(data_dir, 'rgb', '*png')))
        print(f'--------Testing Scene {s}--------')
        
        cnt = 0
        for i, gt_path in tqdm(enumerate(rgb_paths)):
            i_set = int(os.path.split(gt_path)[-1][0])
            if i_set != 2:
                continue
            path = f'{eval_dir}/{cnt}.png'

            gt_img = read_image(gt_path)
            rgb = read_image(path)

            p = -10. * np.log10(np.mean(np.square(rgb - gt_img)))
            psnrs.append(p)
            ssims.append(utils.rgb_ssim(rgb, gt_img, max_val=1))
            lpips_vgg.append(utils.rgb_lpips(rgb, gt_img, net_name='vgg', device='cuda'))
            cnt += 1
        
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing ssim', np.mean(ssims), '(avg)')
        print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')

        with open(f'{eval_dir}_metrics.txt', 'w') as f:
            f.write(f'Testing psnr, {np.mean(psnrs)}\n')
            f.write(f'Testing ssim, {np.mean(ssims)}\n')
            f.write(f'Testing lpips (vgg), {np.mean(lpips_vgg)}\n' )