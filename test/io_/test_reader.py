import unittest

from tensorneko.io import read
from tensorneko_util.io.audio import AudioReader
from tensorneko_util.io.image import ImageReader
from tensorneko_util.io.text import TextReader
from tensorneko_util.io.video import VideoReader
from tensorneko_util.io.matlab import MatReader
from tensorneko_util.io.json import JsonReader


class TestReader(unittest.TestCase):
    """A test for :class:`~tensorneko.io.read`"""

    def test_read(self):
        """Test read object if it has ImageReader, AudioReader, VideoReader, MatReader, JsonReader and TextReader"""
        self.assertIs(read.image, ImageReader)
        self.assertIs(read.audio, AudioReader)
        self.assertIs(read.video, VideoReader)
        self.assertIs(read.text, TextReader)
        self.assertIs(read.mat, MatReader)
        self.assertIs(read.json, JsonReader)
