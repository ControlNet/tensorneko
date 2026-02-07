import unittest

import torch
from torch import Tensor

from tensorneko.layer import GaussianNoise


class TestGaussianNoise(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.GaussianNoise`."""

    def test_train_mode_adds_noise(self):
        """Test that training mode adds noise to input"""
        torch.manual_seed(42)
        x = torch.ones(4, 8)
        noise_layer = GaussianNoise(sigma=0.1, device="cpu")
        noise_layer.train()

        result = noise_layer(x)

        # Check that noise was added (result should differ from input)
        self.assertFalse(torch.equal(result, x))
        # Check that shape is preserved
        self.assertEqual(result.shape, x.shape)
        # Check that values are close but not equal (noise is small)
        self.assertTrue(torch.allclose(result, x, atol=0.5))

    def test_eval_mode_no_noise(self):
        """Test that evaluation mode does not add noise"""
        x = torch.ones(4, 8)
        noise_layer = GaussianNoise(sigma=0.1, device="cpu")
        noise_layer.eval()

        result = noise_layer(x)

        # Check that no noise was added (result should equal input)
        self.assertTrue(torch.equal(result, x))

    def test_sigma_zero_no_noise_train_mode(self):
        """Test that sigma=0 does not add noise even in train mode"""
        x = torch.ones(4, 8)
        noise_layer = GaussianNoise(sigma=0.0, device="cpu")
        noise_layer.train()

        result = noise_layer(x)

        # Check that no noise was added when sigma=0
        self.assertTrue(torch.equal(result, x))

    def test_different_sigma_values(self):
        """Test that different sigma values produce different noise levels"""
        torch.manual_seed(42)
        x = torch.ones(4, 8)

        noise_layer_small = GaussianNoise(sigma=0.01, device="cpu")
        noise_layer_large = GaussianNoise(sigma=0.5, device="cpu")

        noise_layer_small.train()
        noise_layer_large.train()

        result_small = noise_layer_small(x)

        torch.manual_seed(42)  # Reset seed for fair comparison
        result_large = noise_layer_large(x)

        # Calculate differences from original
        diff_small = torch.abs(result_small - x).mean()
        diff_large = torch.abs(result_large - x).mean()

        # Larger sigma should produce larger average difference
        self.assertGreater(diff_large, diff_small)

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape for various inputs"""
        noise_layer = GaussianNoise(sigma=0.1, device="cpu")
        noise_layer.train()

        # Test different shapes
        shapes = [(4, 8), (2, 3, 4), (1, 16, 32, 32)]
        for shape in shapes:
            x = torch.rand(*shape)
            result = noise_layer(x)
            self.assertEqual(result.shape, x.shape)
