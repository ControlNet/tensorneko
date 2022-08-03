import os.path
import unittest

from tensorneko.io import read


class YamlReaderTest(unittest.TestCase):

    def test_read_yaml_as_dict(self):
        obj = read.yaml(os.path.join("test", "resource", "test_read_yaml", "test_read_yaml_as_dict.yaml"))
        self.assertEqual(obj, {"x": 1, "y": 2, "z": "abc"})
