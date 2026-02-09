import unittest
from torch.utils.data import TensorDataset
import torch

from tensorneko.dataset import RoundRobinDataset, NestedDataset


class TestRoundRobinDataset(unittest.TestCase):
    def setUp(self):
        """Set up test datasets."""
        # Create simple tensor datasets
        self.dataset1 = TensorDataset(torch.arange(0, 3))  # [0, 1, 2]
        self.dataset2 = TensorDataset(torch.arange(10, 15))  # [10, 11, 12, 13, 14]

    def test_basic_round_robin(self):
        """Test basic round-robin behavior with padding."""
        dataset = RoundRobinDataset([self.dataset1, self.dataset2], padding=True)

        # With padding=True, length should be max(3, 5) * 2 = 10
        self.assertEqual(len(dataset), 10)

        # Test items alternate between datasets
        # Order: dataset0[0], dataset1[0], dataset0[1], dataset1[1], ...
        item0 = dataset[0]
        item1 = dataset[1]
        self.assertEqual(item0[0].item(), 0)  # dataset1[0]
        self.assertEqual(item1[0].item(), 10)  # dataset2[0]

    def test_round_robin_without_padding(self):
        """Test round-robin behavior without padding."""
        dataset = RoundRobinDataset([self.dataset1, self.dataset2], padding=False)

        # With padding=False, length should be min(3, 5) * 2 = 6
        self.assertEqual(len(dataset), 6)

    def test_custom_order(self):
        """Test custom dataset order."""
        dataset = RoundRobinDataset(
            [self.dataset1, self.dataset2],
            order=[1, 0, 1],  # Alternate: dataset1, dataset0, dataset1
            padding=True,
        )

        # Length should be max(3, 5) * len(order) = 5 * 3 = 15
        # But total_length = each_length * num_datasets = 5 * 2 = 10
        self.assertEqual(len(dataset), 10)

    def test_getitem(self):
        """Test getting items from round-robin dataset."""
        dataset = RoundRobinDataset([self.dataset1, self.dataset2], padding=True)

        # Verify we can get items
        for i in range(len(dataset)):
            item = dataset[i]
            self.assertIsNotNone(item)
            self.assertIsInstance(item[0], torch.Tensor)

    def test_single_dataset(self):
        """Test with single dataset."""
        dataset = RoundRobinDataset([self.dataset1], padding=True)

        # Length should be 3 * 1 = 3
        self.assertEqual(len(dataset), 3)


class SimpleNestedDataset(NestedDataset):
    """Concrete implementation for testing."""

    def load_data(self, index):
        """Load simple tensor datasets based on index."""
        if index == 0:
            return TensorDataset(torch.arange(0, 3))  # 3 items
        elif index == 1:
            return TensorDataset(torch.arange(10, 15))  # 5 items
        else:
            return TensorDataset(torch.arange(100, 102))  # 2 items


class TestNestedDataset(unittest.TestCase):
    def test_basic_nested_dataset(self):
        """Test basic nested dataset functionality."""
        dataset = SimpleNestedDataset(outer_size=2)

        # NestedDataset doesn't implement __len__, but we can manually check _acc
        total_length = dataset._acc[-1]
        self.assertEqual(total_length, 8)  # 3 + 5 = 8

    def test_nested_dataset_getitem(self):
        """Test getting items from nested dataset."""
        dataset = SimpleNestedDataset(outer_size=2)

        # Test that _query_index works correctly
        outer_idx, inner_idx = dataset._query_index(0)
        self.assertEqual(outer_idx, 0)
        self.assertEqual(inner_idx, 0)

        # Test query for item in second dataset
        outer_idx, inner_idx = dataset._query_index(3)
        self.assertEqual(outer_idx, 1)
        self.assertEqual(inner_idx, 0)

    def test_nested_dataset_multiple_outer_size(self):
        """Test nested dataset with multiple outer datasets."""
        dataset = SimpleNestedDataset(outer_size=3)

        # Total length should be 3 + 5 + 2 = 10
        total_length = dataset._acc[-1]
        self.assertEqual(total_length, 10)

        # Test _query_index for items from different datasets
        outer_idx, inner_idx = dataset._query_index(0)
        self.assertEqual(outer_idx, 0)

        outer_idx, inner_idx = dataset._query_index(3)
        self.assertEqual(outer_idx, 1)

        outer_idx, inner_idx = dataset._query_index(8)
        self.assertEqual(outer_idx, 2)


if __name__ == "__main__":
    unittest.main()
