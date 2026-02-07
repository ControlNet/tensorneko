import os
import tempfile
import unittest
from pathlib import Path

from tensorneko_util.io.pickle.pickle_reader import PickleReader
from tensorneko_util.io.pickle.pickle_writer import PickleWriter


class PickleReaderWriterTest(unittest.TestCase):
    def test_roundtrip_dict(self):
        """Roundtrip a dict through pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            data = {"name": "test", "values": [1, 2, 3]}
            PickleWriter.to(path, data)
            result = PickleReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_list(self):
        """Roundtrip a list through pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            data = [1, "two", 3.0, None, True]
            PickleWriter.to(path, data)
            result = PickleReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_custom_object(self):
        """Roundtrip a custom Python object through pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            data = {"nested": {"a": [1, 2, 3]}, "tuple": (4, 5)}
            PickleWriter.to(path, data)
            result = PickleReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_via_constructor(self):
        """Test __new__ aliases for both reader and writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            data = {"key": "value"}
            PickleWriter(path, data)
            result = PickleReader(path)
            self.assertEqual(result, data)

    def test_pathlib_path(self):
        """Test with pathlib.Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.pkl"
            data = [10, 20, 30]
            PickleWriter.to(path, data)
            result = PickleReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_none(self):
        """Roundtrip None through pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            PickleWriter.to(path, None)
            result = PickleReader.of(path)
            self.assertIsNone(result)

    def test_roundtrip_set(self):
        """Roundtrip a set through pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            data = {1, 2, 3}
            PickleWriter.to(path, data)
            result = PickleReader.of(path)
            self.assertEqual(result, data)

    def test_roundtrip_bytes(self):
        """Roundtrip bytes through pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pkl")
            data = b"binary data here"
            PickleWriter.to(path, data)
            result = PickleReader.of(path)
            self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
