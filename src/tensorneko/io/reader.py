from .audio import AudioReader
from .video import VideoReader
from .text import TextReader


class Reader:
    video = VideoReader
    audio = AudioReader
    text = TextReader
