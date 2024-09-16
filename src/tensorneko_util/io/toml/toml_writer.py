import toml
from typing import Union
from pathlib import Path

from .._path_conversion import _path2str


class TomlWriter:

    @classmethod
    def to(cls, path: Union[str, Path], obj: dict):
        """
        Save as Toml file from a dictionary.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            obj (``dict``): The toml data which need to be used for output.
        """
        path = _path2str(path)
        with open(path, "w", encoding="UTF-8") as f:
            toml.dump(obj, f)

    @classmethod
    def __new__(cls, path: Union[str, Path], obj: dict):
        """Alias of :meth:`~tensorneko.io.toml.toml_writer.TomlWriter.to`."""
        path = _path2str(path)
        cls.to(path, obj)
