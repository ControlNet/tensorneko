from typing import Type

from .audio import AudioWriter
from .image import ImageWriter
from .json import JsonWriter
from .pickle import PickleWriter
from .text import TextWriter
from .video import VideoWriter

try:
    from .matlab import MatWriter
    scipy_available = True
except ImportError:
    scipy_available = False
    MatWriter = None

try:
    from .yaml import YamlWriter
    yaml_available = True
except ImportError:
    YamlWriter = object
    yaml_available = False

try:
    from .hdf5 import Hdf5Writer
    h5py_available = True
except ImportError:
    Hdf5Writer = object
    h5py_available = False


class Writer:

    def __init__(self):
        self.image = ImageWriter
        self.text = TextWriter
        self.json = JsonWriter
        self.video = VideoWriter
        self.audio = AudioWriter
        self.pickle = PickleWriter
        self._mat = None
        self._yaml = None
        self._h5 = None

    @property
    def mat(self) -> Type[MatWriter]:
        if scipy_available:
            if self._mat is None:
                self._mat = MatWriter
            return self._mat
        else:
            raise ImportError("Scipy is not available to write mat file.")

    @property
    def yaml(self) -> Type[YamlWriter]:
        if yaml_available:
            if self._yaml is None:
                self._yaml = YamlWriter
            return self._yaml
        else:
            raise ImportError("To use the yaml writer, please install pyyaml.")

    @property
    def h5(self) -> Type[Hdf5Writer]:
        if h5py_available:
            if self._h5 is None:
                self._h5 = Hdf5Writer
            return self._h5
        else:
            raise ImportError("To use the hdf5 writer, please install h5py.")
