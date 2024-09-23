from typing import Union, List

T_Counts = Union[int, List["Counts"]]


class MultiLayerIndexer:
    """
    Multi-layer indexer for hierarchical indexing structures.

    Args:
        counts (``int`` | ``List[int | List]``): A nested structure representing the counts at each layer.
            It can be an integer or a list of counts (which can themselves be lists).

    Examples::

        # Test Case 1
        counts = [5, 10, 15]
        indexer = MultiLayerIndexer(counts)
        print(len(indexer), indexer(12))  # Output: 30 [1, 7]

        # Test Case 2
        counts = [[3, 2], 4, [1, 1, 1]]
        indexer = MultiLayerIndexer(counts)
        print(len(indexer), indexer(8))  # Output: 12 [1, 3]

        # Test Case 3 (Error Handling)
        try:
            counts = [2, [3, [4, 5]]]
            indexer = MultiLayerIndexer(counts)
            print(len(indexer), indexer(20))  # Should raise IndexError
        except IndexError as e:
            print(e)  # Output: Index out of bounds

    """

    def __init__(self, counts: T_Counts):
        self._counts = counts
        self._total = self._total_counts(counts)

    def __call__(self, index: int) -> List[int]:
        """
        Given a global index, return the indices at each layer.

        Args:
            index (``int``): The global index.

        Returns:
            ``List[int]``: A list of indices at each layer corresponding to the global index.

        Raises:
            ``IndexError``: If the provided global index is out of bounds.
        """
        return self._multi_layer_index(self._counts, index)

    @classmethod
    def _multi_layer_index(cls, counts: T_Counts, index: int) -> List[int]:
        """
        A recursive helper function to compute the indices at each layer.

        Args:
            counts (``int`` | ``List[int | List]``): The counts at the current layer.
            index (``int``): The index at the current layer.

        Returns:
            ``list``: A list of indices at each layer.

        Raises:
            ``IndexError``: If the provided index exceeds the available counts.
        """
        if isinstance(counts, int):
            if index < counts:
                return [index]
            else:
                raise IndexError("Index out of bounds")
        elif isinstance(counts, list):
            cumulative_counts = [0]
            for c in counts:
                total_c = cls._total_counts(c)
                cumulative_counts.append(cumulative_counts[-1] + total_c)
            for i in range(len(cumulative_counts) - 1):
                if cumulative_counts[i] <= index < cumulative_counts[i + 1]:
                    sub_idx = index - cumulative_counts[i]
                    sub_indices = cls._multi_layer_index(counts[i], sub_idx)
                    return [i] + sub_indices
            raise IndexError("Index out of bounds")
        else:
            raise ValueError("Invalid counts structure")

    @classmethod
    def _total_counts(cls, counts: T_Counts) -> int:
        """
        Compute the total counts of items under the given counts structure.

        Args:
            counts (``int`` | ``List[int | List]``): The counts structure.

        Returns:
            ``int``: The total count of items.

        Raises:
            ``ValueError``: If the counts structure is invalid.
        """
        if isinstance(counts, int):
            if counts < 0:
                raise ValueError("Counts cannot be negative")
            return counts
        elif isinstance(counts, list):
            return sum(cls._total_counts(c) for c in counts)
        else:
            raise ValueError("Invalid counts structure")

    def __len__(self) -> int:
        return self._total
