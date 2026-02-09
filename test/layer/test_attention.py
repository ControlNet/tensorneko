import unittest

import torch
from torch import Tensor

from tensorneko.layer import ImageAttention, SeqAttention


class TestImageAttention(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.ImageAttention`."""

    def test_forward_shape(self):
        """Test ImageAttention forward pass output shape"""
        B, C, H, W = 2, 4, 3, 3
        x = torch.rand(B, C, H, W)

        layer = ImageAttention(embed_dim=4, num_heads=2)
        result: Tensor = layer(x)

        # Check output shape matches input shape
        self.assertEqual(result.shape, (B, C, H, W))

    def test_return_attention_weights(self):
        """Test ImageAttention returns tuple when return_attention_weights=True"""
        B, C, H, W = 2, 4, 3, 3
        x = torch.rand(B, C, H, W)

        layer = ImageAttention(embed_dim=4, num_heads=2, return_attention_weights=True)
        result = layer(x)

        # Check that result is a tuple with 2 elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check shapes
        output, weights = result
        self.assertEqual(output.shape, (B, C, H, W))
        self.assertEqual(weights.shape, (B, H * W, H * W))


class TestSeqAttention(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.SeqAttention`."""

    def test_forward_shape(self):
        """Test SeqAttention forward pass output shape"""
        B, T, D = 2, 10, 8
        x = torch.rand(B, T, D)

        layer = SeqAttention(embed_dim=8, num_heads=2)
        result: Tensor = layer(x)

        # Check output shape matches input shape
        self.assertEqual(result.shape, (B, T, D))

    def test_return_attention_weights(self):
        """Test SeqAttention returns tuple when return_attention_weights=True"""
        B, T, D = 2, 10, 8
        x = torch.rand(B, T, D)

        layer = SeqAttention(embed_dim=8, num_heads=2, return_attention_weights=True)
        result = layer(x)

        # Check that result is a tuple with 2 elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Check shapes
        output, weights = result
        self.assertEqual(output.shape, (B, T, D))
        self.assertEqual(weights.shape, (B, T, T))
