import unittest

from tensorneko.io import write
from tensorneko_util.io.audio import AudioWriter
from tensorneko_util.io.image import ImageWriter
from tensorneko_util.io.text import TextWriter
from tensorneko_util.io.video import VideoWriter
from tensorneko_util.io.matlab import MatWriter
from tensorneko_util.io.json import JsonWriter


class TestWriter(unittest.TestCase):
    """A test for :class:`~tensorneko.io.read`"""

    def test_write(self):
        """Test read object if it has ImageWriter, AudioWriter, VideoWriter, MatWriter, JsonWriter, and TextWriter"""
        self.assertIs(write.image, ImageWriter)
        self.assertIs(write.audio, AudioWriter)
        self.assertIs(write.video, VideoWriter)
        self.assertIs(write.text, TextWriter)
        self.assertIs(write.mat, MatWriter)
        self.assertIs(write.json, JsonWriter)
