import random
from typing import List, Optional

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from ..util import circular_pad


class RoundRobinDataset(Dataset[T_co]):
    """
    A dataset that combines multiple datasets into one by using a round-robin strategy.

    This class is a subclass of the PyTorch Dataset class and is used to combine multiple datasets.
    It provides the necessary methods for a PyTorch Dataset, such as `__getitem__` and `__len__`.

    Args:
        datasets (``List[Dataset]``): The input datasets.
        order (``List[int]``, optional): The order of the datasets indexes. For example,
            the datasets=[dataset0, dataset1], order=[1, 0, 1], means the data is sampled from
            dataset1, dataset0, dataset1, dataset1, dataset0, dataset1, ...
            Default: [0, 1, 2, ..., n] where `n` is the number of datasets.
        shuffle (``bool``, optional): Whether to shuffle the dataset indexes.
            Default: False.
        padding (``bool``, optional): True will pad the dataset to the longest length, False will cut the dataset to the
            shortest length. Default: True.

    """

    def __init__(self, datasets: List[Dataset[T_co]], order: Optional[List[int]] = None, shuffle: bool = False,
        padding: bool = True
    ):
        super().__init__()
        self.datasets = datasets
        self.order = order or list(range(len(datasets)))  # list of dataset indexes
        self.shuffle = shuffle
        self.padding = padding  # pad or cut

        self.dataset_lengths = [len(dataset) for dataset in datasets]

        # get the length for each sub dataset
        if padding:
            each_length = max(self.dataset_lengths)
        else:
            each_length = min(self.dataset_lengths)

        # total length for this dataset
        self.total_length = each_length * len(datasets)

        # generate the index sampler
        self.samplers = [list(range(length)) for length in self.dataset_lengths]

        if shuffle:
            for sampler in self.samplers:
                random.shuffle(sampler)

        if not padding:
            self.samplers = [sampler[:each_length] for sampler in self.samplers]
        else:
            for i in range(len(self.samplers)):
                self.samplers[i] = circular_pad(self.samplers[i], each_length)

    def __len__(self) -> int:
        """
        Retrieves the total length of the combined dataset.

        Returns:
            The total length of the combined dataset.
        """
        return self.total_length

    def __getitem__(self, index: int) -> T_co:
        """
        Retrieves an item from the combined dataset at a specific index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            The item at the specified index.
        """
        dataset_index = self.order[index % len(self.order)]
        sample_index = self.samplers[dataset_index][index // len(self.order)]
        return self.datasets[dataset_index][sample_index]
