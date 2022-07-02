from .audio import AudioWriter
from .image import ImageWriter
from .json import JsonWriter
from .matlab import MatWriter
from .pickle import PickleWriter
from .text import TextWriter
from .video import VideoWriter


class Writer:

    def __init__(self):
        self.image = ImageWriter
        self.text = TextWriter
        self.json = JsonWriter
        self.video = VideoWriter
        self.audio = AudioWriter
        self.mat = MatWriter
        self.pickle = PickleWriter
