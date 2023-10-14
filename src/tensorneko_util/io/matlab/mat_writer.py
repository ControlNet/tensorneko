from typing import Dict, Any

import scipy.io


class MatWriter:

    @classmethod
    def to(cls, path: str, data: Dict[str, Any]) -> None:
        """
        Write data to a mat file.

        Args:
            path (``str``): path to write.
            data (``Dict[str, Any]``): data to write.
        """
        scipy.io.savemat(path, data)

    def __new__(cls, path: str, data: Dict[str, Any]) -> None:
        """Alias to :meth:`~tensorneko_util.io.mat_writer.MatWriter.to`."""
        cls.to(data, path)
