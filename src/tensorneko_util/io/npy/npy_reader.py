from typing import Union
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile

from .._path_conversion import _path2str


class NpyReader:

    @classmethod
    def of(cls, path: Union[str, Path]) -> Union[np.ndarray, NpzFile]:
        """
        Read numpy array from file.

        Args:
            path (``str`` | ``pathlib.Path``): Path of the numpy file.

        Returns:
            :class:`~numpy.ndarray` | :class:`~numpy.lib.npyio.NpzFile`: The numpy array.
        """
        path = _path2str(path)
        return np.load(path)

    @classmethod
    def of_csc(cls, path: Union[str, Path]) -> np.ndarray:
        """
        Read numpy array from file as CSC sparse matrix.

        Args:
            path (``str`` | ``pathlib.Path``): Path of the numpy file.

        Returns:
            :class:`~numpy.ndarray`: The numpy array.

        """
        path = _path2str(path)
        import scipy.sparse
        return scipy.sparse.load_npz(path).toarray()

    @classmethod
    def of_txt(cls, path: Union[str, Path], delimiter: str = ' ', dtype: type = float) -> np.ndarray:
        """
        Read numpy array from file as text format.

        Args:
            path (``str`` | ``pathlib.Path``): Path of the numpy file.
            delimiter (``str``, optional): The delimiter for each element.
                Default: ' '
            dtype (``type``, optional): The data type for each element.
                Default: float

        Returns:
            :class:`~numpy.ndarray`: The numpy array.
        """
        path = _path2str(path)
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype)

    def __new__(cls, path: Union[str, Path], *args, **kwargs) -> Union[np.ndarray, NpzFile]:
        path = _path2str(path)
        ext = path.split(".")[-1]
        if ext in ("npz", "npy"):
            return cls.of(path)
        elif ext in ("txt", "txt.gz"):
            return cls.of_txt(path, *args, **kwargs)
        else:
            return cls.of(path)
