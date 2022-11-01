from abc import abstractmethod, ABC
from typing import Any, Tuple

import numpy as np
from torch.utils.data.dataset import Dataset


class NestedDataset(Dataset, ABC):

    def __init__(self, outer_size: int):
        super().__init__()
        self.datasets = []
        for i in range(outer_size):
            self.datasets.append(self.load_data(i))

        lengths = np.array([len(dataset) for dataset in self.datasets])
        self._acc = np.cumsum(lengths)

    @abstractmethod
    def load_data(self, index: int) -> Dataset:
        ...

    def _query_index(self, index: int) -> Tuple[int, int]:
        outer_index = np.searchsorted(self._acc, index, "left")
        inner_index = index - self._acc[outer_index - 1]
        return outer_index, inner_index

    def __getitem__(self, index: int) -> Any:
        outer_index, inner_index = self._query_index(index)
        return self.datasets[outer_index][inner_index]