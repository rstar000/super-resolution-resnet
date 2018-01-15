import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from models import resnet

from skimage.io import imread, imsave
from skimage.util import img_as_float

import utils
from sys import argv

border = 30

if len(argv) != 3:
    print('Usage:', argv[0], 'input_file output_file')
    exit(0)


net = resnet.ResNet(32, 128)
net = net.cuda()
net.eval()


def save_model(filename):
    torch.save(net.state_dict(), 'checkpoints/' + filename)


def load_model(filename):
    net.load_state_dict(torch.load(filename))


def add_border(img, n_pixels):
    """
    Add a mirrored border to the image
    """
    h,w,c = img.shape
    img_with_border = np.empty((h+2*n_pixels, w+2*n_pixels, c), np.float32)
    img_with_border[n_pixels:n_pixels+h, n_pixels:n_pixels+w, :] = img
    left_border = img[:, 0:n_pixels, :]
    right_border = img[:, -n_pixels:, :]

    img_with_border[n_pixels:n_pixels+h,0:n_pixels,:] = left_border[:,::-1,:]
    img_with_border[n_pixels:n_pixels+h:,-n_pixels:,:] = right_border[:,::-1,:]

    top_border = img_with_border[n_pixels:2*n_pixels,:,:]
    btm_border = img_with_border[-2*n_pixels - 1:-n_pixels-1,:,:]

    img_with_border[0:n_pixels,:,:] = top_border[::-1,:,:]
    img_with_border[-n_pixels:,:,:] = btm_border[::-1,:,:]

    return img_with_border


def remove_border(img, n_pixels):
    """
    Removes the border
    """
    h,w,c = img.shape

    h -= 2*n_pixels
    w -= 2*n_pixels
    return img[n_pixels:n_pixels+h,n_pixels:n_pixels+w,:]


def upscale_image(img_np):
    """
    Upscales the image. This is not the function you want to use if
    you have a very large image
    :param img_np:
    :return:
    """
    img = add_border(img_np, border)
    img = utils.preprocess_image(img)
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)

    x = torch.from_numpy(img).cuda()
    x = Variable(x, volatile=True, requires_grad=False)

    y = net(x)
    y = y.data[0]
    y = y.cpu().numpy()
    y = utils.restore_image(y)
    y = np.clip(y, 0, 1)
    y = remove_border(y, 2*border)
    return y


def split(img):
    """
    Splits the image into four sections
    """
    h, w, c = img.shape
    x1 = w // 2
    y1 = h // 2

    top_left = img[0:y1,0:x1,:]
    top_right = img[0:y1,x1:w,:]
    btm_left = img[y1:h,0:x1,:]
    btm_right = img[y1:h,x1:w,:]
    return top_left, top_right,btm_left,btm_right


def check_and_upscale(img):
    """
    Recursively split the images and upscale every piece.
    Returns a combined upscaled image
    """
    h, w, c = img.shape
    if h > 600 or w > 600:
        print('Splitting the image! It is too big!')
        split_image = split(img)
        split_res = [check_and_upscale(x) for x in split_image]
        res_image = np.empty((2*h,2*w,c),dtype=np.float32)

        y1,x1,c = split_res[0].shape

        res_image[0:y1, 0:x1,:] = split_res[0]
        res_image[0:y1, x1:2*w,:] = split_res[1]
        res_image[y1:2*h, 0:x1,:] = split_res[2]
        res_image[y1:2*h, x1:2*w,:] = split_res[3]
        return res_image
    else:
        return upscale_image(img)


load_model('checkpoints/2_1.bin')

filename = argv[1]
output = argv[2]

img = imread(filename, as_grey=False)
img = img_as_float(img)


res = check_and_upscale(img)
imsave(output, res)