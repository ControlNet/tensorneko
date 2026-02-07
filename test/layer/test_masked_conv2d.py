import unittest

import torch
from torch import Tensor

from tensorneko.layer import MaskedConv2dA, MaskedConv2dB


class TestMaskedConv2dA(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.MaskedConv2dA`."""

    def test_forward_shape(self):
        """Test MaskedConv2dA forward pass output shape"""
        B, in_channels, H, W = 2, 1, 8, 8
        out_channels = 16
        kernel_size = 3
        x = torch.rand(B, in_channels, H, W)

        layer = MaskedConv2dA(in_channels, out_channels, kernel_size, padding=1)
        result: Tensor = layer(x)

        # Check output shape
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], out_channels)
        self.assertEqual(result.shape[2], H)
        self.assertEqual(result.shape[3], W)

    def test_mask_is_binary(self):
        """Test MaskedConv2dA mask contains only 0s and 1s"""
        in_channels, out_channels = 1, 16
        kernel_size = 3

        layer = MaskedConv2dA(in_channels, out_channels, kernel_size)

        # Check mask is binary
        mask = layer.conv.mask
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))


class TestMaskedConv2dB(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.MaskedConv2dB`."""

    def test_forward_shape(self):
        """Test MaskedConv2dB forward pass output shape"""
        B, in_channels, H, W = 2, 16, 8, 8
        out_channels = 16
        kernel_size = 3
        x = torch.rand(B, in_channels, H, W)

        layer = MaskedConv2dB(in_channels, out_channels, kernel_size, padding=1)
        result: Tensor = layer(x)

        # Check output shape
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], out_channels)
        self.assertEqual(result.shape[2], H)
        self.assertEqual(result.shape[3], W)

    def test_mask_is_binary(self):
        """Test MaskedConv2dB mask contains only 0s and 1s"""
        in_channels, out_channels = 16, 16
        kernel_size = 3

        layer = MaskedConv2dB(in_channels, out_channels, kernel_size)

        # Check mask is binary
        mask = layer.conv.mask
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))

    def test_combined_usage(self):
        """Test MaskedConv2dA and MaskedConv2dB together"""
        B, H, W = 2, 8, 8
        x = torch.rand(B, 1, H, W)

        # First layer: MaskedConv2dA
        layer_a = MaskedConv2dA(1, 16, 3, padding=1)
        result_a: Tensor = layer_a(x)

        # Second layer: MaskedConv2dB
        layer_b = MaskedConv2dB(16, 16, 3, padding=1)
        result_b: Tensor = layer_b(result_a)

        # Check final output shape
        self.assertEqual(result_b.shape[0], B)
        self.assertEqual(result_b.shape[1], 16)
        self.assertEqual(result_b.shape[2], H)
        self.assertEqual(result_b.shape[3], W)
