import numpy as np
import skimage.io as io
from skimage import transform

import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import glob

from tqdm import tqdm_notebook, tqdm

def train_epoch(optimizer, net, train_loader, loss):
    pbar = tqdm_notebook(train_loader)
    for batch in pbar:
        hr, lr = batch
        hr = Variable(hr, requires_grad=False)
        lr = Variable(lr)
        res = net(lr)
        cur_loss = loss(res, hr)

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        
        pbar.set_description("Loss: {}".format(cur_loss.data[0]))


