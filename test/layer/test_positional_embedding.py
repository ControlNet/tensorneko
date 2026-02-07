import unittest

import torch
from torch import Tensor

from tensorneko.layer import PositionalEmbedding, SinCosPositionalEmbedding


class TestPositionalEmbedding(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.PositionalEmbedding`."""

    def test_forward_shape_trainable(self):
        """Test PositionalEmbedding forward pass with trainable=True"""
        B, seq_len, embed_dim = 2, 10, 8
        input_shape = (seq_len, embed_dim)
        x = torch.rand(B, seq_len, embed_dim)

        layer = PositionalEmbedding(input_shape=input_shape, trainable=True)
        result: Tensor = layer(x)

        # Check output shape matches input shape
        self.assertEqual(result.shape, (B, seq_len, embed_dim))

    def test_trainable_flag(self):
        """Test PositionalEmbedding trainable flag sets requires_grad correctly"""
        input_shape = (10, 8)

        # Test trainable=False
        layer = PositionalEmbedding(input_shape=input_shape, trainable=False)
        self.assertFalse(layer.emb.requires_grad)
        self.assertFalse(layer.trainable)

        # Test trainable=True
        layer_trainable = PositionalEmbedding(input_shape=input_shape, trainable=True)
        self.assertTrue(layer_trainable.emb.requires_grad)
        self.assertTrue(layer_trainable.trainable)


class TestSinCosPositionalEmbedding(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.SinCosPositionalEmbedding`."""

    def test_forward_shape(self):
        """Test SinCosPositionalEmbedding forward pass output shape"""
        B, seq_len, embed_dim = 2, 10, 8
        input_shape = (seq_len, embed_dim)
        x = torch.rand(B, seq_len, embed_dim)

        layer = SinCosPositionalEmbedding(input_shape=input_shape)
        result: Tensor = layer(x)

        # Check output shape matches input shape
        self.assertEqual(result.shape, (B, seq_len, embed_dim))

    def test_not_trainable(self):
        """Test SinCosPositionalEmbedding is not trainable by default"""
        input_shape = (10, 8)

        layer = SinCosPositionalEmbedding(input_shape=input_shape)

        # SinCosPositionalEmbedding should not be trainable
        self.assertFalse(layer.emb.requires_grad)
        self.assertFalse(layer.trainable)
