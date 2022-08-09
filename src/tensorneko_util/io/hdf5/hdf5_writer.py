import h5py


class Hdf5Writer:

    @classmethod
    def to(cls, path: str):
        raise NotImplementedError("Hdf5Writer is not implemented yet.")

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Hdf5Writer is not implemented yet.")
