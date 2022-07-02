from .audio import AudioReader
from .image import ImageReader
from .json import JsonReader
from .matlab import MatReader
from .pickle import PickleReader
from .text import TextReader
from .video import VideoReader


class Reader:

    def __init__(self):
        self.image = ImageReader
        self.text = TextReader
        self.json = JsonReader
        self.video = VideoReader
        self.audio = AudioReader
        self.mat = MatReader
        self.pickle = PickleReader
