from typing import Sized

from torch.utils.data.sampler import Sampler, T_co


class SequentialIterSampler(Sampler[T_co]):
    """
    Use to split the large scale data into small subsets for each epochs
    For example, if the dataset size is 1M, and the num_samples = 1000, then each epoch will only use 1000 samples, and
    the next epoch will use the next 1000 samples.

    Args:
        data_source (:class:`torch.utils.data.Dataset`): The dataset to sample from.
        num_samples (``int``): The number of samples to draw from the dataset.

    Examples::

        dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10))
        sampler = SequentialIterSampler(dataset, 100)
        for indexes in sampler:
            print(indexes)
        # [0, 1, 2, ..., 99]
        # [100, 101, 102, ..., 199]
        # [200, 201, 202, ..., 299]
    """

    def __init__(self, data_source: Sized, num_samples: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = num_samples
        self.total_size = len(data_source)
        self.current_position = 0

    def __iter__(self):
        yield from map(lambda x: x % self.total_size,
            range(self.current_position, self.current_position + self.num_samples))
        self.current_position = (self.current_position + self.num_samples) % self.total_size

    def __len__(self):
        return self.num_samples
