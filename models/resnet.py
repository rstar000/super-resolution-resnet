import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import math

def create_conv(n_feat, kernel_size, bias=True):
    return nn.Conv2d(
        n_feat, n_feat, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    '''
    A residual block. The block has two convolutions with padding of 1. The size of resulting images is the same.
    Original images are added to the result (thus it is calles "residual").
    Agruments:
        n_feat - the number of features in a convolution
        kernel_size - the kernel size, as you may guess
        bias - boolean. Use a bias or not?
        bn - boolean. Use batch norm? Preferrably not because it is useless and consumes a lot of memory
        act - the activation function
    '''
    def __init__(
        self, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):


        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(create_conv(n_feat, kernel_size, bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upscale2x(nn.Sequential):
    '''
    An upscaler that uses data from convolutions to reshape the tensor.
    Only argument is the number of features in the previous layer.
    The resulting tensor will have the same number of features, but the images will be twice the size.
    '''
    def __init__(self, n_feat, bias=True):
        modules = []
        modules.append(nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=bias))
        modules.append(nn.PixelShuffle(2))

        super(Upscale2x, self).__init__(*modules)

class ResNet(nn.Module):
    '''
    Enhanced resnet model from EDSR+ paper.
    Uses simplified residual blocks. Upscales the image 2X. The input is an image in range [0,1].
    Arguments:
        n_resblocks: The number of residual blocks in the network
        n_features: The number of features in a convolution. All convolutions have the
                    same, high number of features
    '''
    def __init__(self, n_resblocks, n_features):
        super().__init__()
        self.start = nn.Conv2d(3, n_features, 3, padding=1, bias=True)
        
        resblocks = []
        for i in range(n_resblocks):
            resblocks.append(ResBlock(n_features, 3, res_scale=0.15))
            
        resblocks.append(nn.Conv2d(n_features, n_features, 3, padding=1, bias=True))
        
        upsampler = Upscale2x(n_features)
        
        self.resblocks = nn.Sequential(*resblocks)
        self.upsampler = upsampler
        self.end = nn.Conv2d(n_features, 3, 3, padding=1, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xs = self.start(x)
        x = self.resblocks(xs)
        x = self.upsampler(x + 0.15*xs)
        x = self.end(x)
        return x
        