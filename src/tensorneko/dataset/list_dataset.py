from typing import List

from torch.utils.data.dataset import Dataset, T_co


class ListDataset(Dataset[T_co]):
    """
    A dataset wrapping a list of data.
    """

    def __init__(self, data: List[T_co]):
        super().__init__()
        self.data = data

    def __getitem__(self, index: int) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)
