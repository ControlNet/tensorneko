import unittest

import torch
from torch import Tensor

from tensorneko.layer import Stack


class TestStack(unittest.TestCase):
    """The test class for :class:`tensorneko.layer.Stack`."""

    def test_default_mode_dim_0(self):
        """Test default stack mode with dim=0"""
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        stack = Stack(mode="", dim=0)
        result = stack([x1, x2])
        expected = torch.stack([x1, x2], dim=0)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, (2, 3))

    def test_default_mode_dim_1(self):
        """Test default stack mode with dim=1"""
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        stack = Stack(mode="", dim=1)
        result = stack([x1, x2])
        expected = torch.stack([x1, x2], dim=1)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, (2, 2, 2))

    def test_dstack_mode(self):
        """Test depth stack mode"""
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        stack = Stack(mode="d")
        result = stack([x1, x2])
        expected = torch.dstack([x1, x2])
        self.assertTrue(torch.equal(result, expected))

    def test_vstack_mode(self):
        """Test vertical stack mode"""
        x1 = torch.tensor([[1.0, 2.0, 3.0]])
        x2 = torch.tensor([[4.0, 5.0, 6.0]])
        stack = Stack(mode="v")
        result = stack([x1, x2])
        expected = torch.vstack([x1, x2])
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, (2, 3))

    def test_hstack_mode(self):
        """Test horizontal stack mode"""
        x1 = torch.tensor([[1.0], [2.0]])
        x2 = torch.tensor([[3.0], [4.0]])
        stack = Stack(mode="h")
        result = stack([x1, x2])
        expected = torch.hstack([x1, x2])
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, (2, 2))

    def test_column_mode(self):
        """Test column stack mode"""
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        stack = Stack(mode="column")
        result = stack([x1, x2])
        expected = torch.column_stack([x1, x2])
        self.assertTrue(torch.equal(result, expected))

    def test_row_mode(self):
        """Test row stack mode"""
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        stack = Stack(mode="row")
        result = stack([x1, x2])
        expected = torch.row_stack([x1, x2])
        self.assertTrue(torch.equal(result, expected))

    def test_invalid_mode_raises_value_error(self):
        """Test that invalid mode raises ValueError"""
        with self.assertRaises(ValueError) as context:
            Stack(mode="invalid_mode")
        self.assertIn("Not a valid `mode` argument", str(context.exception))

    def test_mode_and_dim_conflict_raises_assertion_error(self):
        """Test that specifying mode and dim together raises AssertionError"""
        with self.assertRaises(AssertionError) as context:
            Stack(mode="v", dim=1)
        self.assertIn("Other modes cannot specify the dim", str(context.exception))
