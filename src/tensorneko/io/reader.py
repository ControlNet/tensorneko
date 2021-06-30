from .audio import AudioReader
from .video import VideoReader
from .text import TextReader
from .image import ImageReader


class Reader:
    video = VideoReader
    audio = AudioReader
    text = TextReader
    image = ImageReader
