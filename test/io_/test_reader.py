import unittest

from tensorneko.io import read
from tensorneko.io.audio import AudioReader
from tensorneko.io.image import ImageReader
from tensorneko.io.text import TextReader
from tensorneko.io.video import VideoReader


class TestReader(unittest.TestCase):
    """A test for :class:`~tensorneko.io.read`"""

    def test_read(self):
        """Test read object if it has ImageReader, AudioReader, VideoReader and TextReader""" 
        self.assertIs(read.image, ImageReader)
        self.assertIs(read.audio, AudioReader)
        self.assertIs(read.video, VideoReader)
        self.assertIs(read.text, TextReader)
