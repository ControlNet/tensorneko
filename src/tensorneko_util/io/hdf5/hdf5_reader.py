from typing import Union
from pathlib import Path

import h5py

from .._path_conversion import _path2str


class Hdf5Reader:

    @classmethod
    def of(cls, path: Union[str, Path]) -> h5py.File:
        """
        Open a hdf5 file.

        Args:
            path (``str`` | ``pathlib.Path``): Path to the hdf5 file.

        Returns:
            :class:`h5py.File`: The opened hdf5 file.

        """
        path = _path2str(path)
        return h5py.File(path, "r")

    def __new__(cls, path: Union[str, Path]):
        return cls.of(path)
