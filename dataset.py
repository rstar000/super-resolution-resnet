import numpy as np
import skimage.io as io
from skimage import transform

import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob

from skimage.transform import resize, rotate
from skimage.io import imread, imsave
from skimage.util import img_as_float

import random

import utils

rgb_mean = np.asarray([0.5,0.5,0.5], dtype=np.float32)

'''
Data loader for Div2K dataset.
The images are randomly cropped to a set size and then flipped and rotated
'''
class Div2KDataset(data.Dataset):
    def __init__(self, directory, hr_patch_size = 128, patches_per_img = 50):
        self.directory = directory
        self.hr_patch_size = hr_patch_size
        self.patches_per_img = patches_per_img
        file_list = os.listdir(directory)
        self.file_list = []
        for filename in file_list:
            if filename.endswith('.png'):
                self.file_list.append(filename)

        self.cur_file_idx = 0
        self.cur_patch = 0
        cur_filename = self.file_list[self.cur_file_idx]

        self.cur_image = imread(os.path.join(self.directory, cur_filename))
    def __getitem__(self, idx):
        if self.cur_patch == self.patches_per_img:
            self.cur_patch = 0
            self.cur_file_idx = (self.cur_file_idx + 1) % len(self.file_list)
            cur_filename  = self.file_list[self.cur_file_idx]
            self.cur_image = imread(os.path.join(self.directory, cur_filename))

        patch = self.get_patch()
        patch = self.augment(patch)
        hr, lr = self.get_tensor_pair(patch)
        self.cur_patch += 1

        return self.to_tensor(hr), self.to_tensor(lr)

    def __len__(self):
        return len(self.file_list) * self.patches_per_img

    '''
    Cut a random patch from an image. The image must be in a H*W*C format
    '''
    def get_patch(self):
        patch_size = self.hr_patch_size
        h,w,c = self.cur_image.shape

        w0 = random.randint(0, w - patch_size - 1)
        h0 = random.randint(0, h - patch_size - 1)

        patch = self.cur_image[h0:h0+patch_size, w0:w0+patch_size, :]
        return patch.copy()

    '''
    Generates ready to use tensors from a patch. Downscales a patch to create a LR image.
    Returns a pair [HR, LR]
    '''
    def get_tensor_pair(self, patch):
        patch_size = self.hr_patch_size
        lr_patch_size = patch_size / 2

        lr_patch = resize(patch, [lr_patch_size, lr_patch_size, 3], mode='reflect')

        hr_patch = utils.preprocess_image(patch)
        lr_patch = utils.preprocess_image(lr_patch)

        return hr_patch, lr_patch

    '''
    Augments the patch by using random rotations and flips
    '''
    def augment(self, patch):
        rotations = [0,90,180,270]

        angle = rotations[random.randint(0,3)]

        patch = rotate(patch, angle)
        if(random.choice([True, False])):
            patch = np.fliplr(patch)

        if(random.choice([True, False])):
            patch = np.flipud(patch)

        return patch

    def to_tensor(self, np_image):
        np_image = np_image.astype(np.float32)
        res_tensor = torch.from_numpy(np_image.copy()).cuda()
        return res_tensor

