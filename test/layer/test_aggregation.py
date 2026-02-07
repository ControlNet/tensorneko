import unittest

import torch
from torch import Tensor

from tensorneko.layer import Aggregation


class TestAggregation(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.Aggregation`."""

    def test_mean_no_dim(self):
        """Test mean aggregation without specifying dim"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="mean", dim=None)
        result = agg(x)
        expected = torch.mean(x)
        self.assertTrue(torch.allclose(result, expected))

    def test_mean_dim_1(self):
        """Test mean aggregation with dim=1"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="mean", dim=1)
        result = agg(x)
        expected = torch.tensor([2.0, 5.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_mean_dim_tuple(self):
        """Test mean aggregation with dim=(1,2)"""
        x = torch.rand(2, 3, 4, 5)
        agg = Aggregation(mode="mean", dim=(1, 2))
        result = agg(x)
        expected = torch.mean(x, dim=(1, 2))
        self.assertTrue(torch.allclose(result, expected))
        self.assertEqual(result.shape, (2, 5))

    def test_sum_no_dim(self):
        """Test sum aggregation without specifying dim"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="sum", dim=None)
        result = agg(x)
        expected = torch.sum(x)
        self.assertTrue(torch.allclose(result, expected))

    def test_sum_dim_1(self):
        """Test sum aggregation with dim=1"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="sum", dim=1)
        result = agg(x)
        expected = torch.tensor([6.0, 15.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_max_no_dim(self):
        """Test max aggregation with dim=0"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="max", dim=0)
        result = agg(x)
        expected = torch.max(x, dim=0)
        self.assertTrue(torch.equal(result.values, expected.values))
        self.assertTrue(torch.equal(result.indices, expected.indices))

    def test_max_dim_1(self):
        """Test max aggregation with dim=1"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="max", dim=1)
        result = agg(x)
        expected = torch.max(x, dim=1)
        self.assertTrue(torch.equal(result.values, expected.values))
        self.assertTrue(torch.equal(result.indices, expected.indices))

    def test_min_no_dim(self):
        """Test min aggregation with dim=0"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="min", dim=0)
        result = agg(x)
        expected = torch.min(x, dim=0)
        self.assertTrue(torch.equal(result.values, expected.values))
        self.assertTrue(torch.equal(result.indices, expected.indices))

    def test_min_dim_1(self):
        """Test min aggregation with dim=1"""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agg = Aggregation(mode="min", dim=1)
        result = agg(x)
        expected = torch.min(x, dim=1)
        self.assertTrue(torch.equal(result.values, expected.values))
        self.assertTrue(torch.equal(result.indices, expected.indices))

    def test_invalid_mode_raises_value_error(self):
        """Test that invalid mode raises ValueError"""
        with self.assertRaises(ValueError) as context:
            Aggregation(mode="invalid_mode")
        self.assertIn("Wrong mode value", str(context.exception))
