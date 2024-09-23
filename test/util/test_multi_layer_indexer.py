import unittest

from tensorneko_util.util.multi_layer_indexer import MultiLayerIndexer


class UtilMultiLayerIndexerTest(unittest.TestCase):
    """Unit tests for the MultiLayerIndexer class."""

    def test_case_1(self):
        """Test Case 1: Simple flat structure."""
        counts = [5, 10, 15]
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 30, "Total count should be 30.")
        self.assertEqual(indexer(12), [1, 7], "Index 12 should map to [1, 7].")
        self.assertEqual(indexer(0), [0, 0], "Index 0 should map to [0, 0].")
        self.assertEqual(indexer(29), [2, 14], "Index 29 should map to [2, 14].")

    def test_case_2(self):
        """Test Case 2: Nested structure with multiple layers."""
        counts = [[3, 2], 4, [1, 1, 1]]
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 12, "Total count should be 12.")
        self.assertEqual(indexer(8), [1, 3], "Index 8 should map to [1, 3].")
        self.assertEqual(indexer(0), [0, 0, 0], "Index 0 should map to [0, 0, 0].")
        self.assertEqual(indexer(11), [2, 2, 0], "Index 11 should map to [2, 2, 0].")

    def test_case_3_error_handling(self):
        """Test Case 3: Handling out-of-bounds indices."""
        counts = [2, [3, [4, 5]]]
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 2 + 3 + 4 + 5, "Total count should be 14.")
        with self.assertRaises(IndexError, msg="Index 14 should raise IndexError."):
            indexer(14)
        with self.assertRaises(IndexError, msg="Index 20 should raise IndexError."):
            indexer(20)
        with self.assertRaises(IndexError, msg="Negative index should raise IndexError."):
            indexer(-1)

    def test_empty_counts(self):
        """Test handling of empty counts."""
        counts = []
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 0, "Total count should be 0.")
        with self.assertRaises(IndexError, msg="Any index should raise IndexError for empty counts."):
            indexer(0)

    def test_single_layer(self):
        """Test with a single integer count."""
        counts = 10
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 10, "Total count should be 10.")
        self.assertEqual(indexer(5), [5], "Index 5 should map to [5].")
        with self.assertRaises(IndexError, msg="Index 10 should raise IndexError."):
            indexer(10)

    def test_deeply_nested_counts(self):
        """Test with a deeply nested counts structure."""
        counts = [[[2, 3], 4], 5]
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), (2 + 3) + 4 + 5, "Total count should be 14.")
        self.assertEqual(indexer(0), [0, 0, 0, 0], "Index 0 should map to [0, 0, 0, 0].")
        self.assertEqual(indexer(4), [0, 0, 1, 2], "Index 4 should map to [0, 0, 1, 2].")
        self.assertEqual(indexer(7), [0, 1, 2], "Index 7 should map to [0, 1, 2].")

    def test_non_integer_counts(self):
        """Test with invalid counts structure."""
        counts = [2, "invalid", 3]
        with self.assertRaises(ValueError, msg="Non-integer counts should raise ValueError."):
            MultiLayerIndexer(counts)

    def test_negative_counts(self):
        """Test with negative counts."""
        counts = [5, -10, 15]
        with self.assertRaises(ValueError, msg="Negative counts should raise ValueError."):
            MultiLayerIndexer(counts)

    def test_zero_counts(self):
        """Test with zero counts."""
        counts = [0, 5, 0]
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 5, "Total count should be 5.")
        self.assertEqual(indexer(0), [1, 0], "Index 0 should map to [1, 0].")
        self.assertEqual(indexer(4), [1, 4], "Index 4 should map to [1, 4].")
        with self.assertRaises(IndexError, msg="Index 5 should raise IndexError."):
            indexer(5)

    def test_large_index(self):
        """Test with a very large index."""
        counts = [1000, 2000, 3000]
        indexer = MultiLayerIndexer(counts)
        self.assertEqual(len(indexer), 6000, "Total count should be 6000.")
        self.assertEqual(indexer(5999), [2, 2999], "Index 5999 should map to [2, 2999].")
        with self.assertRaises(IndexError, msg="Index 6000 should raise IndexError."):
            indexer(6000)
