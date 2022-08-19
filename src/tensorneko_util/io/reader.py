from typing import Type

from .audio import AudioReader
from .image import ImageReader
from .json import JsonReader
from .pickle import PickleReader
from .text import TextReader
from .video import VideoReader

try:
    from .matlab import MatReader
    scipy_available = True
except ImportError:
    scipy_available = False
    MatReader = None

try:
    from .yaml import YamlReader
    yaml_available = True
except ImportError:
    YamlReader = object
    yaml_available = False

try:
    from .hdf5 import Hdf5Reader
    h5py_available = True
except ImportError:
    Hdf5Reader = object
    h5py_available = False


class Reader:

    def __init__(self):
        self.image = ImageReader
        self.text = TextReader
        self.json = JsonReader
        self.video = VideoReader
        self.audio = AudioReader
        self.pickle = PickleReader
        self._mat = None
        self._yaml = None
        self._h5 = None

    @property
    def mat(self) -> Type[MatReader]:
        if scipy_available:
            if self._mat is None:
                self._mat = MatReader
            return self._mat
        else:
            raise ImportError("Scipy is not available to read mat file.")

    @property
    def yaml(self) -> Type[YamlReader]:
        if yaml_available:
            if self._yaml is None:
                self._yaml = YamlReader
            return self._yaml
        else:
            raise ImportError("To use the yaml reader, please install pyyaml.")

    @property
    def h5(self) -> Type[Hdf5Reader]:
        if h5py_available:
            if self._h5 is None:
                self._h5 = Hdf5Reader
            return self._h5
        else:
            raise ImportError("To use the hdf5 reader, please install h5py.")
