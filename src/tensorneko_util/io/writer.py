from .image import ImageWriter
from .json import JsonWriter
from .text import TextWriter


class Writer:

    def __init__(self):
        self.image = ImageWriter
        self.text = TextWriter
        self.json = JsonWriter
