import unittest

import torch
from torch import Tensor

from tensorneko.module.gated_conv import GatedConv


class TestGatedConv(unittest.TestCase):
    def test_init_mask_type_a(self):
        model = GatedConv("A", in_channels=4, kernel_size=3, padding=1)
        self.assertFalse(model.residual is None)

    def test_init_mask_type_b(self):
        model = GatedConv("B", in_channels=4, kernel_size=3, padding=1)
        self.assertFalse(model.residual is None)

    def test_init_invalid_mask_type(self):
        with self.assertRaises(ValueError):
            GatedConv("C", in_channels=4, kernel_size=3, padding=1)

    def test_activation(self):
        x = torch.rand(2, 8, 4, 4)
        out = GatedConv.activation(x)
        self.assertEqual(out.shape, torch.Size([2, 4, 4, 4]))

    def test_forward_mask_a_with_residual(self):
        model = GatedConv("A", in_channels=4, kernel_size=3, padding=1, residual=True)
        x_v = torch.rand(2, 4, 8, 8)
        x_h = torch.rand(2, 4, 8, 8)
        h_out, v_out = model(x_v, x_h)
        self.assertEqual(h_out.shape, torch.Size([2, 4, 8, 8]))
        self.assertEqual(v_out.shape, torch.Size([2, 4, 8, 8]))

    def test_forward_mask_b_with_residual(self):
        model = GatedConv("B", in_channels=4, kernel_size=3, padding=1, residual=True)
        x_v = torch.rand(2, 4, 8, 8)
        x_h = torch.rand(2, 4, 8, 8)
        h_out, v_out = model(x_v, x_h)
        self.assertEqual(h_out.shape, torch.Size([2, 4, 8, 8]))
        self.assertEqual(v_out.shape, torch.Size([2, 4, 8, 8]))

    def test_forward_without_residual(self):
        model = GatedConv("A", in_channels=4, kernel_size=3, padding=1, residual=False)
        x_v = torch.rand(2, 4, 8, 8)
        x_h = torch.rand(2, 4, 8, 8)
        h_out, v_out = model(x_v, x_h)
        self.assertEqual(h_out.shape, torch.Size([2, 4, 8, 8]))
        self.assertEqual(v_out.shape, torch.Size([2, 4, 8, 8]))

    def test_forward_returns_tuple(self):
        model = GatedConv("B", in_channels=4, kernel_size=3, padding=1)
        x_v = torch.rand(2, 4, 8, 8)
        x_h = torch.rand(2, 4, 8, 8)
        result = model(x_v, x_h)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
