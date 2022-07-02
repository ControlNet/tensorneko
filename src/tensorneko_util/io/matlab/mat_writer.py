from typing import Dict, Any

import scipy.io


class MatWriter:

    @classmethod
    def to(cls, data: Dict[str, Any], path: str) -> None:
        """
        Write data to a mat file.

        Args:
            data (``Dict[str, Any]``): data to write.
            path (``str``): path to write.
        """
        scipy.io.savemat(path, data)

    def __new__(cls, data: Dict[str, Any], path: str) -> None:
        """Alias to :meth:`~tensorneko_util.io.mat_writer.MatWriter.to`."""
        cls.to(data, path)
