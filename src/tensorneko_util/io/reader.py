from typing import Type

from .audio import AudioReader
from .image import ImageReader
from .json import JsonReader
from .matlab import MatReader
from .pickle import PickleReader
from .text import TextReader
from .video import VideoReader

try:
    from .yaml import YamlReader
    yaml_available = True
except ImportError:
    YamlReader = object
    yaml_available = False


class Reader:

    def __init__(self):
        self.image = ImageReader
        self.text = TextReader
        self.json = JsonReader
        self.video = VideoReader
        self.audio = AudioReader
        self.mat = MatReader
        self.pickle = PickleReader
        self._yaml = None

    @property
    def yaml(self) -> Type[YamlReader]:
        if yaml_available:
            if self._yaml is None:
                from .yaml import YamlReader
                self._yaml = YamlReader
            return self._yaml
        else:
            raise ImportError("To use the yaml reader, please install pyyaml.")
