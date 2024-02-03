from typing import List

from torch.utils.data.dataset import Dataset, T_co


class ListDataset(Dataset[T_co]):
    """
    A dataset wrapping a list of data.

    This class is a subclass of the PyTorch Dataset class and is used to wrap a list of data.
    It provides the necessary methods for a PyTorch Dataset, such as __getitem__ and __len__.

    Args:
        data (``List[T_co]``): The list of data that the dataset wraps.
    """

    def __init__(self, data: List[T_co]):
        """
        The constructor for the ListDataset class.

        Args:
            data (``List[T_co]``): The list of data that the dataset will wrap.
        """
        super().__init__()
        self.data = data

    def __getitem__(self, index: int) -> T_co:
        """
        Retrieves an item from the dataset at a specific index.

        Args:
            index (``int``): The index of the item to retrieve.

        Returns:
            The item at the specified index.
        """
        return self.data[index]

    def __len__(self):
        """
        Retrieves the length of the dataset.

        Returns:
            The length of the dataset (i.e., the number of items in the dataset).
        """
        return len(self.data)
