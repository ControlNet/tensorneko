from typing import Union

import numpy as np
from numpy.lib.npyio import NpzFile


class NpyReader:

    @classmethod
    def of(cls, path: str) -> Union[np.ndarray, NpzFile]:
        """
        Read numpy array from file.

        Args:
            path (``str``): Path of the numpy file.

        Returns:
            :class:`~numpy.ndarray` | :class:`~numpy.lib.npyio.NpzFile`: The numpy array.
        """
        return np.load(path)

    @classmethod
    def of_csc(cls, path: str) -> np.ndarray:
        """
        Read numpy array from file as CSC sparse matrix.

        Args:
            path (``str``): Path of the numpy file.

        Returns:
            :class:`~numpy.ndarray`: The numpy array.

        """
        import scipy.sparse
        return scipy.sparse.load_npz(path).toarray()

    @classmethod
    def of_txt(cls, path: str, delimiter: str = ' ', dtype: type = float) -> np.ndarray:
        """
        Read numpy array from file as text format.

        Args:
            path (``str``): Path of the numpy file.
            delimiter (``str``, optional): The delimiter for each element.
                Default: ' '
            dtype (``type``, optional): The data type for each element.
                Default: float

        Returns:
            :class:`~numpy.ndarray`: The numpy array.
        """
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype)

    @classmethod
    def __new__(cls, path: str, *args, **kwargs) -> Union[np.ndarray, NpzFile]:
        ext = path.split(".")[-1]
        if ext in ("npz", "npy"):
            return cls.of(path)
        elif ext in ("txt", "txt.gz"):
            return cls.of_txt(path, *args, **kwargs)
        else:
            return cls.of(path)
