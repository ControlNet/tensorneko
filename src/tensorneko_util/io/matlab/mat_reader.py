from typing import Dict, Any, Union
from pathlib import Path

import scipy.io

from .._path_conversion import _path2str


class MatReader:

    @classmethod
    def of(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read a mat file.

        Args:
            path (``str`` | ``pathlib.Path``): The path of the mat file.
        """
        path = _path2str(path)
        return scipy.io.loadmat(path)

    def __new__(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Alias to :meth:`~tensorneko_util.io.matlab.MatReader.of`."""
        return cls.of(path)
