from .image import ImageReader
from .json import JsonReader
from .text import TextReader


class Reader:
    def __init__(self):
        self.image = ImageReader
        self.text = TextReader
        self.json = JsonReader
