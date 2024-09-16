from typing import Union
from pathlib import Path

import numpy as np

from .._path_conversion import _path2str


class NpyWriter:

    @classmethod
    def to(cls, path: Union[str, Path], arr: np.ndarray) -> None:
        """
        Save numpy array to file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            arr (:class:`~numpy.ndarray`): The numpy array for output.
        """
        path = _path2str(path)
        np.save(path, arr)

    @classmethod
    def to_csc(cls, path: Union[str, Path], arr: np.ndarray) -> None:
        """
        Save numpy array to file as CSC sparse matrix.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            arr (:class:`~numpy.ndarray`): The numpy array for output.
        """
        path = _path2str(path)
        import scipy.sparse
        scipy.sparse.save_npz(path, scipy.sparse.csc_matrix(arr))

    @classmethod
    def to_npz(cls, path: Union[str, Path], compressed: bool = False, **kwargs) -> None:
        """
        Save numpy array to file as npz format.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            compressed (``bool``, optional): The flag for compressed npz file.
                Default: False
            **kwargs: The numpy arrays for output.
        """
        path = _path2str(path)
        if compressed:
            np.savez_compressed(path, **kwargs)
        else:
            np.savez(path, **kwargs)

    @classmethod
    def to_txt(cls, path: Union[str, Path], arr: np.ndarray, delimiter: str = ' ', newline: str = '\n') -> None:
        """
        Save numpy array to file as text format.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            arr (:class:`~numpy.ndarray`): The numpy array for output.
            delimiter (``str``, optional): The delimiter for each element.
                Default: ' '
            newline (``str``, optional): The newline for each row.
                Default: '\n'
        """
        path = _path2str(path)
        np.savetxt(path, arr, delimiter=delimiter, newline=newline)

    def __new__(cls, path: Union[str, Path], *args, **kwargs) -> None:
        path = _path2str(path)
        ext = path.split(".")[-1]
        if ext == "npy":
            return cls.to(path, *args, **kwargs)
        elif ext == "npz":
            return cls.to_npz(path, *args, **kwargs)
        elif ext in ("txt", "txt.gz"):
            return cls.to_txt(path, *args, **kwargs)
        else:
            raise ValueError(f"Unknown file extension: {ext}")
