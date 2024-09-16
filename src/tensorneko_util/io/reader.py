from typing import Type, Union, Any
from pathlib import Path

from ._path_conversion import _path2str
from .audio import AudioReader
from .image import ImageReader
from .json import JsonReader
from .npy import NpyReader
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

try:
    from .toml import TomlReader
    toml_available = True
except ImportError:
    TomlReader = object
    toml_available = False


class Reader:

    def __init__(self):
        self.image = ImageReader
        self.text = TextReader
        self.json = JsonReader
        self.npy = NpyReader
        self.video = VideoReader
        self.audio = AudioReader
        self.pickle = PickleReader
        self._mat = None
        self._yaml = None
        self._h5 = None
        self._toml = None

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

    @property
    def toml(self) -> Type[TomlReader]:
        if toml_available:
            if self._toml is None:
                self._toml = TomlReader
            return self._toml
        else:
            raise ImportError("To use the toml reader, please install toml.")

    def __call__(self, path: Union[str, Path], *args, **kwargs) -> Any:
        """Automatically infer the file type and return the corresponding result. """
        path = _path2str(path)

        if path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png") or path.endswith(".bmp"):
            return self.image(path, *args, **kwargs)
        elif path.endswith(".txt"):
            return self.text(path, *args, **kwargs)
        elif path.endswith(".json"):
            return self.json(path, *args, **kwargs)
        elif path.endswith(".npy") or path.endswith(".npz"):
            return self.npy(path, *args, **kwargs)
        elif path.endswith(".mp4") or path.endswith(".avi") or path.endswith(".mov") or path.endswith(".mkv"):
            return self.video(path, *args, **kwargs)
        elif path.endswith(".wav") or path.endswith(".mp3") or path.endswith(".flac"):
            return self.audio(path, *args, **kwargs)
        elif path.endswith(".pkl") or path.endswith(".pickle"):
            assert len(args) == 0 and len(kwargs) == 0, "Pickle reader does not support extra arguments."
            return self.pickle(path)
        elif path.endswith(".mat"):
            return self.mat(path, *args, **kwargs)
        elif path.endswith(".yaml") or path.endswith(".yml"):
            return self.yaml(path, *args, **kwargs)
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            assert len(args) == 0 and len(kwargs) == 0, "Hdf5 reader does not support extra arguments."
            return self.h5(path)
        elif path.endswith(".toml"):
            assert len(args) == 0 and len(kwargs) == 0, "Toml reader does not support extra arguments."
            return self.toml(path)
        else:
            raise ValueError("Unknown file type: {}".format(path))
