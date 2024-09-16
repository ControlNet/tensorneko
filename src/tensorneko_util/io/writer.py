from typing import Type, Union
from pathlib import Path

from ._path_conversion import _path2str
from .audio import AudioWriter
from .image import ImageWriter
from .json import JsonWriter
from .npy import NpyWriter
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

try:
    from .toml import TomlWriter
    toml_available = True
except ImportError:
    TomlWriter = object
    toml_available = False


class Writer:

    def __init__(self):
        self.image = ImageWriter
        self.text = TextWriter
        self.json = JsonWriter
        self.npy = NpyWriter
        self.video = VideoWriter
        self.audio = AudioWriter
        self.pickle = PickleWriter
        self._mat = None
        self._yaml = None
        self._h5 = None
        self._toml = None

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

    @property
    def toml(self) -> Type[TomlWriter]:
        if toml_available:
            if self._toml is None:
                self._toml = TomlWriter
            return self._toml
        else:
            raise ImportError("To use the toml writer, please install toml.")

    def __call__(self, path: Union[str, Path], obj, *args, **kwargs):
        """Automatically infer the file type and return the corresponding result. """
        path = _path2str(path)

        if path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png") or path.endswith(".bmp"):
            self.image(path, obj, *args, **kwargs)
        elif path.endswith(".txt"):
            return self.text(path, obj, *args, **kwargs)
        elif path.endswith(".json"):
            return self.json(path, obj, *args, **kwargs)
        elif path.endswith(".npy") or path.endswith(".npz"):
            return self.npy(path, obj, *args, **kwargs)
        elif path.endswith(".mp4") or path.endswith(".avi") or path.endswith(".mov") or path.endswith(".mkv"):
            return self.video(path, obj, *args, **kwargs)
        elif path.endswith(".wav") or path.endswith(".mp3") or path.endswith(".flac"):
            return self.audio(path, obj, *args, **kwargs)
        elif path.endswith(".pkl") or path.endswith(".pickle"):
            assert len(args) == 0 and len(kwargs) == 0, "Pickle reader does not support extra arguments."
            return self.pickle(path, obj)
        elif path.endswith(".mat"):
            assert len(args) == 0 and len(kwargs) == 0, "Mat reader does not support extra arguments."
            return self.mat(path, obj)
        elif path.endswith(".yaml") or path.endswith(".yml"):
            assert len(args) == 0 and len(kwargs) == 0, "Yaml reader does not support extra arguments."
            return self.yaml(path, obj)
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            raise NotImplementedError("Hdf5 writer is not implemented yet.")
        elif path.endswith(".toml"):
            assert len(args) == 0 and len(kwargs) == 0, "Toml reader does not support extra arguments."
            return self.toml(path, obj)
        else:
            raise ValueError("Unknown file type: {}".format(path))
