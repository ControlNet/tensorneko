from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np
from torch.utils.data.dataset import Dataset, T_co


class NestedDataset(Dataset[T_co], ABC):
    """
    A dataset that wraps multiple datasets.

    This class is a subclass of the PyTorch Dataset class and is used to wrap multiple datasets.
    It provides the necessary methods for a PyTorch Dataset, such as `__getitem__` and `load_data`.

    Args:
        outer_size (``int``): The number of datasets to be wrapped.

    Attributes:
        datasets (``List[Dataset[T_co]]``): The list of datasets that the NestedDataset wraps.
    """

    def __init__(self, outer_size: int):
        super().__init__()
        self.datasets = []
        for i in range(outer_size):
            self.datasets.append(self.load_data(i))

        lengths = np.array([len(dataset) for dataset in self.datasets])
        self._acc = np.cumsum(lengths)

    @abstractmethod
    def load_data(self, index: int) -> Dataset[T_co]:
        """
        Abstract method to load data for a specific index.

        Args:
            index (``int``): The index of the dataset to load.

        Returns:
            The loaded dataset.
        """
        ...

    def _query_index(self, index: int) -> Tuple[int, int]:
        """
        Retrieves the outer and inner indices for a specific index.

        Args:
            index (``int``): The index to query.

        Returns:
            A tuple containing the outer and inner indices.
        """
        outer_index = np.searchsorted(self._acc, index, "left")
        inner_index = index - self._acc[outer_index - 1]
        return outer_index, inner_index

    def __getitem__(self, index: int) -> T_co:
        """
        Retrieves an item from the nested dataset at a specific index.

        Args:
            index (``int``): The index of the item to retrieve.

        Returns:
            The item at the specified index.
        """
        outer_index, inner_index = self._query_index(index)
        return self.datasets[outer_index][inner_index]
