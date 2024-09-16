import pickle
from typing import Union
from pathlib import Path

from .._path_conversion import _path2str


class PickleWriter:

    @classmethod
    def to(cls, path: Union[str, Path], obj: object) -> None:
        """
        Save the object to a file.

        Args:
            path (``str`` | ``pathlib.Path``): The path to the file.
            obj (``object``): The object to save.
        """
        path = _path2str(path)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def __new__(cls, path: Union[str, Path], obj: object) -> None:
        """Alias to :meth:`~tensorneko_util.io.pickle.PickleWriter.to`."""
        path = _path2str(path)
        return cls.to(path, obj)
