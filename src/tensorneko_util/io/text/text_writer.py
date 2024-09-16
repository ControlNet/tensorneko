from typing import Union
from pathlib import Path

from .._path_conversion import _path2str


class TextWriter:
    """TextWriter for writing files for text"""

    @staticmethod
    def to_plain(path: Union[str, Path], text: str, encoding: str = "UTF-8") -> None:
        """
        Save as a plain text file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            text (``str``): The content for output.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
        """
        path = _path2str(path)
        with open(path, "w", encoding=encoding) as file:
            file.write(text)

    to = to_plain

    def __new__(cls, path: Union[str, Path], text: str, encoding: str = "UTF-8") -> None:
        """Alias of :meth:`~TextWriter.to`"""
        path = _path2str(path)
        cls.to(path, text, encoding)
