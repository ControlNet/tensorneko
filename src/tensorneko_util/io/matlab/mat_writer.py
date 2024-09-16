from typing import Dict, Any, Union
from pathlib import Path

import scipy.io

from .._path_conversion import _path2str


class MatWriter:

    @classmethod
    def to(cls, path: Union[str, Path], data: Dict[str, Any]) -> None:
        """
        Write data to a mat file.

        Args:
            path (``str`` | ``pathlib.Path``): path to write.
            data (``Dict[str, Any]``): data to write.
        """
        path = _path2str(path)
        scipy.io.savemat(path, data)

    def __new__(cls, path: Union[str, Path], data: Dict[str, Any]) -> None:
        """Alias to :meth:`~tensorneko_util.io.mat_writer.MatWriter.to`."""
        cls.to(data, path)
