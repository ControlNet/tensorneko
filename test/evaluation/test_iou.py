import unittest

import torch
import numpy as np

from tensorneko.evaluation import iou_1d, iou_2d


class TestIou1D(unittest.TestCase):
    def test_iou_1d_perfect_overlap(self):
        """Test perfect overlap should return 1.0"""
        proposal = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[0.0, 1.0]])

        iou = iou_1d(proposal, target)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)

    def test_iou_1d_no_overlap(self):
        """Test no overlap should return 0.0"""
        proposal = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[2.0, 3.0]])

        iou = iou_1d(proposal, target)
        self.assertAlmostEqual(iou.item(), 0.0, places=5)

    def test_iou_1d_partial_overlap(self):
        """Test partial overlap with known value"""
        proposal = torch.tensor([[0.0, 2.0]])
        target = torch.tensor([[1.0, 3.0]])

        # intersection: [1.0, 2.0] -> length = 1.0
        # union: [0.0, 3.0] -> length = 3.0
        # iou = 1.0 / 3.0 = 0.333...
        iou = iou_1d(proposal, target)
        self.assertAlmostEqual(iou.item(), 1.0 / 3.0, places=5)

    def test_iou_1d_multi_proposals_targets(self):
        """Test M x N shape with multiple proposals and targets"""
        proposal = torch.tensor([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
        target = torch.tensor([[0.5, 2.5], [3.0, 5.0]])

        iou = iou_1d(proposal, target)
        # Should return shape [3, 2]
        self.assertEqual(iou.shape, torch.Size([3, 2]))

        # Test specific values
        # proposal[0] vs target[0]: [0.0, 2.0] vs [0.5, 2.5]
        # intersection: [0.5, 2.0] -> 1.5
        # union: [0.0, 2.5] -> 2.5
        # iou = 1.5 / 2.5 = 0.6
        self.assertAlmostEqual(iou[0, 0].item(), 0.6, places=5)

    def test_iou_1d_numpy_input(self):
        """Test with numpy array input"""
        proposal = np.array([[0.0, 1.0]])
        target = np.array([[0.0, 1.0]])

        iou = iou_1d(proposal, target)
        self.assertIsInstance(iou, torch.Tensor)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)


class TestIou2D(unittest.TestCase):
    def test_iou_2d_perfect_overlap(self):
        """Test perfect overlap should return 1.0"""
        proposal = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])

        iou = iou_2d(proposal, target)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)

    def test_iou_2d_no_overlap(self):
        """Test no overlap should return 0.0"""
        proposal = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target = torch.tensor([[2.0, 2.0, 3.0, 3.0]])

        iou = iou_2d(proposal, target)
        self.assertAlmostEqual(iou.item(), 0.0, places=5)

    def test_iou_2d_partial_overlap(self):
        """Test partial overlap with known value"""
        proposal = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        target = torch.tensor([[1.0, 1.0, 3.0, 3.0]])

        # intersection: [1.0, 1.0, 2.0, 2.0] -> area = 1.0
        # proposal area: 4.0
        # target area: 4.0
        # union: 4.0 + 4.0 - 1.0 = 7.0
        # iou = 1.0 / 7.0 = 0.142857...
        iou = iou_2d(proposal, target)
        self.assertAlmostEqual(iou.item(), 1.0 / 7.0, places=5)

    def test_iou_2d_multi_proposals_targets(self):
        """Test M x N shape with multiple proposals and targets"""
        proposal = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]]
        )
        target = torch.tensor([[0.5, 0.5, 2.5, 2.5], [3.0, 3.0, 5.0, 5.0]])

        iou = iou_2d(proposal, target)
        # Should return shape [N, M] = [2, 3] based on implementation
        self.assertEqual(iou.shape, torch.Size([2, 3]))

        # Test specific values
        # target[0] vs proposal[0]: [0.5, 0.5, 2.5, 2.5] vs [0.0, 0.0, 2.0, 2.0]
        # intersection: [0.5, 0.5, 2.0, 2.0] -> area = 1.5 * 1.5 = 2.25
        # proposal area: 4.0
        # target area: 4.0
        # union: 4.0 + 4.0 - 2.25 = 5.75
        # iou = 2.25 / 5.75 = 0.391304...
        self.assertAlmostEqual(iou[0, 0].item(), 2.25 / 5.75, places=5)

    def test_iou_2d_numpy_input(self):
        """Test with numpy array input"""
        proposal = np.array([[0.0, 0.0, 2.0, 2.0]])
        target = np.array([[0.0, 0.0, 2.0, 2.0]])

        iou = iou_2d(proposal, target)
        self.assertIsInstance(iou, torch.Tensor)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
