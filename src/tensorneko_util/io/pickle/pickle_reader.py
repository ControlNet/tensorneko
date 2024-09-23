import pickle
from typing import Union, Any
from pathlib import Path

from .._path_conversion import _path2str


class PickleReader:

    @classmethod
    def of(cls, path: Union[str, Path]) -> Any:
        """
        Save the object to a file.

        Args:
            path (``str`` | ``pathlib.Path``): The path to the file.
        """
        path = _path2str(path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __new__(cls, path: Union[str, Path]) -> Any:
        """Alias to :meth:`~tensorneko_util.io.pickle.PickleReader.of`."""
        path = _path2str(path)
        return cls.of(path)
