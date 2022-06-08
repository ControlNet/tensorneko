import unittest

from torch import Tensor

from tensorneko.layer import Conv2d
from tensorneko.util import F
import torch.nn as nn
import torch


class TestConv2d(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.Conv2d`."""

    @property
    def b(self):
        return 8

    @property
    def h(self):
        return 32

    @property
    def w(self):
        return 32

    @property
    def c(self):
        return self.in_features

    @property
    def in_features(self):
        return 3

    @property
    def out_features(self):
        return 64

    @property
    def kernel_size(self):
        return 3, 3

    @property
    def stride(self):
        return 1, 1

    @property
    def padding(self):
        return 1, 1

    @property
    def activation_factory(self):
        return nn.ReLU

    @property
    def normalization_factory(self):
        return F(nn.BatchNorm2d, self.out_features)

    def test_single_layer(self):
        """The single layer without batch normalization and activation"""
        # Create a batch of size 8
        x = torch.rand(self.b, self.c, self.h, self.w)

        # Create a single layer
        neko_layer = Conv2d(self.in_features, self.out_features, self.kernel_size, self.stride, self.padding)
        torch_layer = nn.Sequential(neko_layer.conv)

        # Forward prop
        neko_result: Tensor = neko_layer(x)
        pytorch_result: Tensor = torch_layer(x)

        # Check the output
        self.assertTrue((neko_result == pytorch_result).all())

    def test_single_layer_with_activation(self):
        """A single layer with activation"""
        # Create a batch of size 8
        x = torch.rand(self.b, self.c, self.h, self.w)

        # Create a single layer
        neko_layer = Conv2d(self.in_features, self.out_features, self.kernel_size, self.stride, self.padding, build_activation=self.activation_factory)
        torch_layer = nn.Sequential(neko_layer.conv, neko_layer.activation)

        # Forward prop
        neko_result: Tensor = neko_layer(x)
        pytorch_result: Tensor = torch_layer(x)

        # Check the output
        self.assertTrue((neko_result == pytorch_result).all())

    def test_single_layer_with_normalization(self):
        """A single layer with batch normalization"""
        # Create a batch of size 8
        x = torch.rand(self.b, self.c, self.h, self.w)

        # Create a single layer
        neko_layer = Conv2d(self.in_features, self.out_features, self.kernel_size, self.stride, self.padding, build_normalization=self.normalization_factory)
        torch_layer = nn.Sequential(neko_layer.conv, neko_layer.normalization)

        # Forward prop
        neko_result: Tensor = neko_layer(x)
        pytorch_result: Tensor = torch_layer(x)

        # Check the output
        self.assertTrue((neko_result == pytorch_result).all())

    def test_single_layer_with_normalization_and_activation(self):
        """A single layer with batch normalization and activation"""
        # Create a batch of size 8
        x = torch.rand(self.b, self.c, self.h, self.w)

        # Create a single layer
        neko_layer = Conv2d(self.in_features, self.out_features, self.kernel_size, self.stride, self.padding, build_activation=self.activation_factory, build_normalization=self.normalization_factory)
        torch_layer = nn.Sequential(neko_layer.conv, neko_layer.normalization, neko_layer.activation)

        # Forward prop
        neko_result: Tensor = neko_layer(x)
        pytorch_result: Tensor = torch_layer(x)

        # Check the output
        self.assertTrue((neko_result == pytorch_result).all())

