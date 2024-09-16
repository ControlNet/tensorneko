import toml
from typing import Union
from pathlib import Path

from .._path_conversion import _path2str


class TomlReader:

    @classmethod
    def of(cls, path: Union[str, Path]) -> dict:
        """
        Open a toml file.

        Args:
            path (``str`` | ``pathlib.Path``): Path to the toml file.

        Returns:
            ``dict``: The opened toml file.

        """
        path = _path2str(path)
        return toml.load(path)

    def __new__(cls, path: Union[str, Path]):
        path = _path2str(path)
        return cls.of(path)
