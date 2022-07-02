from typing import Dict, Any

import scipy.io


class MatReader:

    @classmethod
    def of(cls, path: str) -> Dict[str, Any]:
        """
        Read a mat file.

        Args:
            path (``str``): The path of the mat file.
        """
        return scipy.io.loadmat(path)

    def __new__(cls, path: str) -> Dict[str, Any]:
        """Alias to :meth:`~tensorneko_util.io.matlab.MatReader.of`."""
        return cls.of(path)
