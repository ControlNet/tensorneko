import h5py


class Hdf5Reader:

    @classmethod
    def of(cls, path: str) -> h5py.File:
        """
        Open a hdf5 file.

        Args:
            path (``str``): Path to the hdf5 file.

        Returns:
            :class:`h5py.File`: The opened hdf5 file.

        """
        return h5py.File(path, "r")

    def __new__(cls, path: str):
        return cls.of(path)
