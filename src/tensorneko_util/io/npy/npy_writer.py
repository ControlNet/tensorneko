import numpy as np


class NpyWriter:

    @classmethod
    def to(cls, path: str, arr: np.ndarray) -> None:
        """
        Save numpy array to file.

        Args:
            path (``str``): The path of output file.
            arr (:class:`~numpy.ndarray`): The numpy array for output.
        """
        np.save(path, arr)

    @classmethod
    def to_csc(cls, path: str, arr: np.ndarray) -> None:
        """
        Save numpy array to file as CSC sparse matrix.

        Args:
            path (``str``): The path of output file.
            arr (:class:`~numpy.ndarray`): The numpy array for output.
        """
        import scipy.sparse
        scipy.sparse.save_npz(path, scipy.sparse.csc_matrix(arr))

    @classmethod
    def to_npz(cls, path: str, compressed: bool = False, **kwargs) -> None:
        """
        Save numpy array to file as npz format.

        Args:
            path (``str``): The path of output file.
            compressed (``bool``, optional): The flag for compressed npz file.
                Default: False
            **kwargs: The numpy arrays for output.
        """
        if compressed:
            np.savez_compressed(path, **kwargs)
        else:
            np.savez(path, **kwargs)

    @classmethod
    def to_txt(cls, path: str, arr: np.ndarray, delimiter: str = ' ', newline: str = '\n') -> None:
        """
        Save numpy array to file as text format.

        Args:
            path (``str``): The path of output file.
            arr (:class:`~numpy.ndarray`): The numpy array for output.
            delimiter (``str``, optional): The delimiter for each element.
                Default: ' '
            newline (``str``, optional): The newline for each row.
                Default: '\n'
        """
        np.savetxt(path, arr, delimiter=delimiter, newline=newline)

    @classmethod
    def __new__(cls, path: str, *args, **kwargs) -> None:
        ext = path.split(".")[-1]
        if ext == "npy":
            return cls.to(path, *args, **kwargs)
        elif ext == "npz":
            return cls.to_npz(path, *args, **kwargs)
        elif ext in ("txt", "txt.gz"):
            return cls.to_txt(path, *args, **kwargs)
        else:
            raise ValueError(f"Unknown file extension: {ext}")
