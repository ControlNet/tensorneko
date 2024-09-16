from typing import Union
from pathlib import Path

from .._path_conversion import _path2str


class TextReader:
    """TextReader for reading text file"""

    @staticmethod
    def of_plain(path: Union[str, Path], encoding: str = "UTF-8") -> str:
        """
        Read texts of a file.

        Args:
            path (``str`` | ``pathlib.Path``): Text file path.
            encoding (``str``, optional): File encoding. Default "UTF-8".

        Returns:
            ``str``: The texts of given file.
        """
        path = _path2str(path)
        with open(path, "r", encoding=encoding) as file:
            text = file.read()
        return text

    of = of_plain

    def __new__(cls, path: Union[str, Path], encoding: str = "UTF-8") -> str:
        """Alias of :meth:`~TextReader.of"""
        path = _path2str(path)
        return cls.of(path, encoding)
