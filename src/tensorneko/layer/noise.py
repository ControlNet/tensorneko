from typing import Union

import torch

from .. import NekoModule
from ..util import Device


class GaussianNoise(NekoModule):
    """
    The layer for adding a Gaussian noise. Get inspired from YannDubs1 (2020).

    Args:
        sigma (``float``, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector. Default ``0.1``.
        device (:class:``tensorneko.util.type.Device``): The model running device. Default ``"cuda"``.

    References:
        Writing a simple Gaussian noise layer in Pytorch. (2017). Retrieved
        from https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/3
    """

    def __init__(self, sigma=0.1, device: Union[Device, str] = "cuda"):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach()
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x
