from typing import Type

from .audio import AudioWriter
from .image import ImageWriter
from .json import JsonWriter
from .matlab import MatWriter
from .pickle import PickleWriter
from .text import TextWriter
from .video import VideoWriter

try:
    from .yaml import YamlWriter

    yaml_available = True
except ImportError:
    YamlWriter = object
    yaml_available = False


class Writer:

    def __init__(self):
        self.image = ImageWriter
        self.text = TextWriter
        self.json = JsonWriter
        self.video = VideoWriter
        self.audio = AudioWriter
        self.mat = MatWriter
        self.pickle = PickleWriter
        self._yaml = None

    @property
    def yaml(self) -> Type[YamlWriter]:
        if yaml_available:
            if self._yaml is None:
                from .yaml import YamlWriter
                self._yaml = YamlWriter
            return self._yaml
        else:
            raise ImportError("To use the yaml writer, please install pyyaml.")
