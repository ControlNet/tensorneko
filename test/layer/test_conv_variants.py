import unittest

import torch
import torch.nn as nn
from torch import Tensor

from tensorneko.layer import Conv1d, Conv3d


class TestConv1d(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.Conv1d`."""

    def test_forward_shape(self):
        """Test Conv1d forward pass output shape"""
        B, in_channels, seq_len = 2, 8, 20
        out_channels = 16
        kernel_size = 3
        x = torch.rand(B, in_channels, seq_len)

        layer = Conv1d(in_channels, out_channels, kernel_size)
        result: Tensor = layer(x)

        # Check output shape
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], out_channels)

    def test_with_activation_and_normalization(self):
        """Test Conv1d with activation and normalization"""
        B, in_channels, seq_len = 2, 8, 20
        out_channels = 16
        kernel_size = 3
        x = torch.rand(B, in_channels, seq_len)

        layer = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            build_activation=nn.ReLU,
            build_normalization=lambda: nn.BatchNorm1d(out_channels),
            normalization_after_activation=True,
        )
        result: Tensor = layer(x)

        # Check output shape
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], out_channels)


class TestConv3d(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.Conv3d`."""

    def test_forward_shape(self):
        """Test Conv3d forward pass output shape"""
        B, in_channels, D, H, W = 2, 3, 8, 8, 8
        out_channels = 16
        kernel_size = 3
        x = torch.rand(B, in_channels, D, H, W)

        layer = Conv3d(in_channels, out_channels, kernel_size)
        result: Tensor = layer(x)

        # Check output shape
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], out_channels)

    def test_with_activation_and_normalization(self):
        """Test Conv3d with activation and normalization"""
        B, in_channels, D, H, W = 2, 3, 8, 8, 8
        out_channels = 16
        kernel_size = 3
        x = torch.rand(B, in_channels, D, H, W)

        layer = Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            build_activation=nn.ReLU,
            build_normalization=lambda: nn.BatchNorm3d(out_channels),
            normalization_after_activation=True,
        )
        result: Tensor = layer(x)

        # Check output shape
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], out_channels)
