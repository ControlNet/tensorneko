import os
import tempfile
import unittest
from pathlib import Path

from tensorneko_util.io.yaml.yaml_reader import YamlReader
from tensorneko_util.io.yaml.yaml_writer import YamlWriter


class YamlWriterTest(unittest.TestCase):
    def test_write_dict(self):
        """Write a dict to YAML and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.yaml")
            data = {"name": "test", "value": 42}
            YamlWriter.to(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)

    def test_write_nested_dict(self):
        """Write a nested dict to YAML and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.yaml")
            data = {"server": {"host": "localhost", "port": 8080}, "debug": True}
            YamlWriter.to(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)

    def test_write_list(self):
        """Write a list to YAML and read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.yaml")
            data = [1, 2, 3, "four"]
            YamlWriter.to(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)

    def test_write_via_constructor(self):
        """Test __new__ alias for YamlWriter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.yaml")
            data = {"key": "value"}
            YamlWriter(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)

    def test_pathlib_path(self):
        """Test YamlWriter with pathlib.Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.yaml"
            data = {"x": 1, "y": 2, "z": "abc"}
            YamlWriter.to(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_complex_data(self):
        """Roundtrip complex nested data through YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.yaml")
            data = {
                "database": {"host": "localhost", "port": 5432, "name": "mydb"},
                "features": ["auth", "logging"],
                "enabled": True,
                "count": 100,
            }
            YamlWriter.to(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)

    def test_write_none_value(self):
        """Write dict with None values to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.yaml")
            data = {"key": None}
            YamlWriter.to(path, data)
            result = YamlReader.of(path)
            self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
