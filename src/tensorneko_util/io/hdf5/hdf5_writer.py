from typing import Union
from pathlib import Path

import h5py


class Hdf5Writer:

    @classmethod
    def to(cls, path: Union[str, Path]):
        raise NotImplementedError("Hdf5Writer is not implemented yet.")

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Hdf5Writer is not implemented yet.")
