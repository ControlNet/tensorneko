import os.path
import unittest

from tensorneko.io import read


class TestTextReader(unittest.TestCase):
    def test_read(self):
        """Test the TextReader to read the file in resources directory"""
        text = read.text.of(os.path.join("test", "resource", "test_read_text", "text.txt"))
        self.assertEqual(text, """This is a sample text file.
Second line.""")
