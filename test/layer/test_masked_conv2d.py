import unittest

import torch
from torch import Tensor

from tensorneko.layer import MaskedConv2dA, MaskedConv2dB
from tensorneko.layer.masked_conv2d import (
    _VerticalStackConv2dA,
    _VerticalStackConv2dB,
    _HorizontalStackConv2dA,
    _HorizontalStackConv2dB,
)


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


class TestMaskedConv2dClassNames(unittest.TestCase):
    def test_masked_conv2d_a_class_name(self):
        layer = MaskedConv2dA(1, 4, 3)
        self.assertEqual(layer._class_name, "tensorneko.layer.MaskedConv2dA")

    def test_masked_conv2d_b_class_name(self):
        layer = MaskedConv2dB(1, 4, 3)
        self.assertEqual(layer._class_name, "tensorneko.layer.MaskedConv2dB")


class TestVerticalStackConv2dA(unittest.TestCase):
    def test_mask_is_binary(self):
        layer = _VerticalStackConv2dA(1, 4, 3, padding=1)
        self.assertTrue(torch.all((layer.mask == 0) | (layer.mask == 1)))

    def test_mask_zeros_bottom_half(self):
        layer = _VerticalStackConv2dA(1, 4, 3, padding=1)
        _, _, kh, _ = layer.weight.size()
        self.assertTrue(torch.all(layer.mask[:, :, kh // 2 :] == 0))

    def test_forward_shape(self):
        layer = _VerticalStackConv2dA(1, 4, 3, padding=1)
        x = torch.rand(2, 1, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, torch.Size([2, 4, 8, 8]))


class TestVerticalStackConv2dB(unittest.TestCase):
    def test_mask_is_binary(self):
        layer = _VerticalStackConv2dB(1, 4, 3, padding=1)
        self.assertTrue(torch.all((layer.mask == 0) | (layer.mask == 1)))

    def test_mask_zeros_below_center_plus_one(self):
        layer = _VerticalStackConv2dB(1, 4, 3, padding=1)
        _, _, kh, _ = layer.weight.size()
        self.assertTrue(torch.all(layer.mask[:, :, kh // 2 + 1 :] == 0))

    def test_forward_shape(self):
        layer = _VerticalStackConv2dB(1, 4, 3, padding=1)
        x = torch.rand(2, 1, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, torch.Size([2, 4, 8, 8]))


class TestHorizontalStackConv2dA(unittest.TestCase):
    def test_mask_is_binary(self):
        layer = _HorizontalStackConv2dA(1, 4, 3, padding=1)
        self.assertTrue(torch.all((layer.mask == 0) | (layer.mask == 1)))

    def test_mask_zeros_right_half(self):
        layer = _HorizontalStackConv2dA(1, 4, 3, padding=1)
        _, _, _, kw = layer.weight.size()
        self.assertTrue(torch.all(layer.mask[:, :, :, kw // 2 :] == 0))

    def test_forward_shape(self):
        layer = _HorizontalStackConv2dA(1, 4, 3, padding=1)
        x = torch.rand(2, 1, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, torch.Size([2, 4, 8, 8]))


class TestHorizontalStackConv2dB(unittest.TestCase):
    def test_mask_is_binary(self):
        layer = _HorizontalStackConv2dB(1, 4, 3, padding=1)
        self.assertTrue(torch.all((layer.mask == 0) | (layer.mask == 1)))

    def test_mask_zeros_right_of_center_plus_one(self):
        layer = _HorizontalStackConv2dB(1, 4, 3, padding=1)
        _, _, _, kw = layer.weight.size()
        self.assertTrue(torch.all(layer.mask[:, :, :, kw // 2 + 1 :] == 0))

    def test_forward_shape(self):
        layer = _HorizontalStackConv2dB(1, 4, 3, padding=1)
        x = torch.rand(2, 1, 8, 8)
        out = layer(x)
        self.assertEqual(out.shape, torch.Size([2, 4, 8, 8]))
