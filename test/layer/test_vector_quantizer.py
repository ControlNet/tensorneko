import unittest

import torch
from torch import Tensor

from tensorneko.layer import VectorQuantizer


class TestVectorQuantizer(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.VectorQuantizer`."""

    def test_forward_output_shapes(self):
        """Test that forward returns correct output shapes"""
        vq = VectorQuantizer(n_embeddings=8, embedding_dim=4)
        z = torch.randn(2, 4, 3, 3)

        z_q, embedding_loss = vq(z)

        # z_q should have same shape as input z
        self.assertEqual(z_q.shape, (2, 4, 3, 3))
        # embedding_loss should be a scalar
        self.assertEqual(embedding_loss.ndim, 0)

    def test_embedding_loss_is_finite(self):
        """Test that embedding loss is finite"""
        vq = VectorQuantizer(n_embeddings=8, embedding_dim=4)
        z = torch.randn(2, 4, 3, 3)

        z_q, embedding_loss = vq(z)

        self.assertTrue(torch.isfinite(embedding_loss))

    def test_predict_indexes_range(self):
        """Test that predict_indexes returns valid index range"""
        n_embeddings = 8
        vq = VectorQuantizer(n_embeddings=n_embeddings, embedding_dim=4)
        z = torch.randn(2, 4, 3, 3)

        z_index = vq.predict_indexes(z)

        # All indexes should be in range [0, n_embeddings-1]
        self.assertTrue((z_index >= 0).all())
        self.assertTrue((z_index < n_embeddings).all())

    def test_predict_indexes_shape(self):
        """Test that predict_indexes returns correct shape"""
        vq = VectorQuantizer(n_embeddings=8, embedding_dim=4)
        z = torch.randn(2, 4, 3, 3)

        z_index = vq.predict_indexes(z)

        # z_index should be (batch, height, width)
        self.assertEqual(z_index.shape, (2, 3, 3))

    def test_indexing_shape(self):
        """Test that indexing returns correct shape"""
        vq = VectorQuantizer(n_embeddings=8, embedding_dim=4)
        z_index = torch.randint(0, 8, (2, 3, 3))

        z_q = vq.indexing(z_index)

        # z_q should be (batch, channels, height, width)
        self.assertEqual(z_q.shape, (2, 4, 3, 3))

    def test_forward_quantization_consistency(self):
        """Test that forward quantization is consistent with predict_indexes and indexing"""
        vq = VectorQuantizer(n_embeddings=8, embedding_dim=4)
        z = torch.randn(2, 4, 3, 3)

        # Get quantized output from forward
        z_q_forward, _ = vq(z)

        # Get quantized output by predict_indexes and indexing separately
        z_index = vq.predict_indexes(z)
        z_q_manual = vq.indexing(z_index)

        # The quantized values should be very close
        # Note: forward has gradient tracking, so we compare detached values
        self.assertTrue(torch.allclose(z_q_forward.detach(), z_q_manual, atol=1e-6))

    def test_different_beta_values(self):
        """Test that different beta values affect embedding loss"""
        z = torch.randn(2, 4, 3, 3)

        vq_small_beta = VectorQuantizer(n_embeddings=8, embedding_dim=4, beta=0.1)
        vq_large_beta = VectorQuantizer(n_embeddings=8, embedding_dim=4, beta=1.0)

        # Copy embeddings to make comparison fair
        vq_large_beta.embedding.weight.data = (
            vq_small_beta.embedding.weight.data.clone()
        )

        _, loss_small = vq_small_beta(z)
        _, loss_large = vq_large_beta(z)

        # Losses should differ when beta differs
        self.assertNotEqual(loss_small.item(), loss_large.item())

    def test_gradient_flow(self):
        """Test that gradients flow through the quantization layer"""
        vq = VectorQuantizer(n_embeddings=8, embedding_dim=4)
        z = torch.randn(2, 4, 3, 3, requires_grad=True)

        z_q, embedding_loss = vq(z)

        # Compute a simple loss and backprop
        loss = z_q.sum() + embedding_loss
        loss.backward()

        # Check that gradients exist for input
        self.assertIsNotNone(z.grad)
        self.assertTrue((z.grad != 0).any())
