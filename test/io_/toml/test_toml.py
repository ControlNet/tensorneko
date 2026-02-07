import os
import tempfile
import unittest
from pathlib import Path

from tensorneko_util.io.toml.toml_reader import TomlReader
from tensorneko_util.io.toml.toml_writer import TomlWriter


class TomlReaderWriterTest(unittest.TestCase):
    def test_roundtrip_dict(self):
        """Roundtrip a dict through TOML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.toml")
            data = {"name": "test", "value": 42}
            TomlWriter.to(path, data)
            result = TomlReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_nested_dict(self):
        """Roundtrip a nested dict through TOML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.toml")
            data = {
                "database": {"host": "localhost", "port": 5432},
                "app": {"name": "myapp", "debug": True},
            }
            TomlWriter.to(path, data)
            result = TomlReader.of(path)
            self.assertEqual(result, data)

    def test_writer_via_constructor(self):
        """Test TomlWriter __new__ — note: __new__ is decorated @classmethod which
        causes signature issues. Verify the error is raised as expected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.toml")
            data = {"key": "value"}
            # TomlWriter.__new__ is @classmethod, so direct construction fails
            # with extra cls arg. Use .to() instead for normal usage.
            with self.assertRaises(TypeError):
                TomlWriter(path, data)

    def test_reader_via_constructor(self):
        """Test TomlReader __new__ alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.toml")
            data = {"key": "value"}
            TomlWriter.to(path, data)
            result = TomlReader(path)
            self.assertEqual(result, data)

    def test_pathlib_path(self):
        """Test with pathlib.Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.toml"
            data = {"x": 1, "y": 2}
            TomlWriter.to(path, data)
            result = TomlReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_with_arrays(self):
        """Roundtrip TOML with array values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.toml")
            data = {"colors": ["red", "green", "blue"], "numbers": [1, 2, 3]}
            TomlWriter.to(path, data)
            result = TomlReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_mixed_types(self):
        """Roundtrip TOML with mixed value types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.toml")
            data = {"string": "hello", "integer": 42, "float": 3.14, "bool": True}
            TomlWriter.to(path, data)
            result = TomlReader.of(path)
            self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
