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

from matplotlib import pyplot as plt

rgb_mean = np.asarray([0.0,  0.0,  0.0])

'''
preprocess_image

Convert a regular image into a 3d tensor and subtracts the mean value
'''
def preprocess_image(img):
    img = img_as_float(img)
    img -= rgb_mean
    img = img.transpose(2,0,1)
    return img

'''
Restore an image from a 3D tensor
'''
def restore_image(tensor):
    x = tensor.transpose(1,2,0)
    x += rgb_mean
    return x


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
