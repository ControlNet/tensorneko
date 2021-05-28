# TODO: need refactor

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import init


def upscale_2d(x):
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


def downscale_2d(x):
    # since the factor is same for height and width the bilinear downsampling is just like an avg pool
    return F.avg_pool2d(x, 2, 2)  # downscales in factor 2


def compute_r1_gradient_penalty(d_result_real, real_images):
    real_grads = torch.autograd.grad(d_result_real.sum(), real_images, create_graph=True, retain_graph=True)[0]
    # real_grads = torch.autograd.grad(d_result_real, real_images,
    #                                  grad_outputs=torch.ones_like(d_result_real),
    #                                  create_graph=True, retain_graph=True)[0]
    r1_penalty = 0.5 * torch.sum(real_grads.pow(2.0), dim=[1, 2, 3]) # Norm on all dims but batch

    return r1_penalty

class StyleInstanceNorm2d(nn.Module):
    def __init__(self, channels):
        super(StyleInstanceNorm2d, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)

    def forward(self, x):
        # TODO: implement  nn.InstanceNorm2d here to save redundant mean/var computation
        assert(len(x.shape) == 4)
        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style = torch.cat((m, std), dim=1)

        x = self.instance_norm(x)

        return x, style


class LearnablePreScaleBlur(nn.Module):
    """
    From StyleFan paper:
    "We replace the nearest-neighbor up/downsampling in both net-works with bilinear sampling, which we implement by
    low-pass filtering the activations with a separable 2nd order bi-nomial filter after each upsampling layer and
    before each downsampling layer [Making Convolutional Networks Shift-Invariant Again]."
    """
    def __init__(self, channels):
        super(LearnablePreScaleBlur, self).__init__()
        f = np.array([[1, 2, 1]])
        f = np.matmul(f.transpose(), f)
        f = f / np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)
        return x


def pixel_norm(x, epsilon=1e-8):
    return x / torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)
    # return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


class StyleAffineTransform(nn.Module):
    '''
    The 'A' unit in StyleGan:
    Learned affine transform A, this module is used to transform the midiate vector w into a style vector.
    it outputs a mean and std scalar for each channel for a given number of channels.
    this mean and std are used in AdaIn as facror and bias to shift the norm of the input channels
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = LREQ_FC_Layer(dim_latent, n_channel * 2)
        # "the biases associated with mean that we initialize to one"
        self.transform.bias.data[:n_channel] = 1
        self.transform.bias.data[n_channel:] = 0

    def forward(self, w):
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


class NoiseScaler(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    'B' unit in SttleGan
    '''

    def __init__(self, n_channel):
        super().__init__()
        # per channel wiehgt
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        result = noise * self.weight
        return result


class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    Shift the mean and variance of each chnnel in the input image by a given factor and bias
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1) # split number of chenels to two
        normed_input = self.norm(image)
        rescaled_image = normed_input * factor + bias
        return rescaled_image


class LREQ_FC_Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LREQ_FC_Layer, self).__init__(in_features, out_features, bias)
        init.normal_(self.weight)
        if bias:
            init.zeros_(self.bias)
        self.c = np.sqrt(2.0) / np.sqrt(in_features)

    def forward(self, input):
        return F.linear(input,
                        self.weight * self.c,
                        self.bias)



class Lreq_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, output_padding=0, dilation=1, bias=True):
        super(Lreq_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.dilation = (dilation, dilation)

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        init.normal_(self.weight)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.c = np.sqrt(2.0) / np.sqrt(np.prod(self.kernel_size) * in_channels)

    def __str__(self):
        return f"LreqConv2d({self.in_channels}, {self.out_channels}, k={self.kernel_size[0]}, p={self.padding[0]})"


    def forward(self, x):
        return F.conv2d(x,
                        self.weight * self.c,
                        self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation
                        )