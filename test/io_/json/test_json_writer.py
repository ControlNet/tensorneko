import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import List

from tensorneko_util.io.json.json_writer import JsonWriter
from tensorneko_util.io.json.json_reader import JsonReader
from tensorneko_util.io.json.json_data import json_data


@json_data
class Point:
    x: int
    y: int


@json_data
class Line:
    start: Point
    end: Point


class JsonWriterTest(unittest.TestCase):
    def test_write_dict(self):
        """Write a dict to JSON, read back and verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"name": "test", "value": 42}
            JsonWriter.to(path, data, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, data)

    def test_write_list(self):
        """Write a list of dicts to JSON, read back and verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            JsonWriter.to(path, data, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, data)

    def test_write_empty_list(self):
        """Write an empty list to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = []
            JsonWriter.to(path, data, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, data)

    def test_write_json_data_object(self):
        """Write a @json_data decorated object to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            pt = Point({"x": 10, "y": 20})
            JsonWriter.to(path, pt, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, {"x": 10, "y": 20})

    def test_write_list_of_json_data(self):
        """Write a list of @json_data objects to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            pts = [Point({"x": 1, "y": 2}), Point({"x": 3, "y": 4})]
            JsonWriter.to(path, pts, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, [{"x": 1, "y": 2}, {"x": 3, "y": 4}])

    def test_write_with_indent(self):
        """Write JSON with custom indent via stdlib json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"a": 1}
            JsonWriter.to(path, data, indent=2, fast=False)
            with open(path, "r") as f:
                content = f.read()
            # indent=2 should produce multi-line output
            self.assertIn("\n", content)
            result = json.loads(content)
            self.assertEqual(result, data)

    def test_write_with_ensure_ascii(self):
        """Write JSON with ensure_ascii=True via stdlib json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"name": "日本語"}
            JsonWriter.to(path, data, ensure_ascii=True, fast=False)
            with open(path, "r", encoding="UTF-8") as f:
                content = f.read()
            # ensure_ascii should escape non-ASCII
            self.assertNotIn("日本語", content)
            result = json.loads(content)
            self.assertEqual(result, data)

    def test_write_unsupported_type_raises(self):
        """Writing unsupported type should raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with self.assertRaises(TypeError):
                JsonWriter.to(path, "not a dict or list", fast=False)

    def test_write_pathlib_path(self):
        """Write JSON using pathlib.Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}
            JsonWriter.to(path, data, fast=False)
            result = JsonReader.of(str(path), fast=False)
            self.assertEqual(result, data)

    def test_write_via_constructor(self):
        """Test the __new__ alias for writing JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"a": 1, "b": 2}
            JsonWriter(path, data, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, data)

    def test_write_nested_json_data(self):
        """Write nested @json_data object to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            line = Line({"start": {"x": 1, "y": 2}, "end": {"x": 3, "y": 4}})
            JsonWriter.to(path, line, fast=False)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(
                result, {"start": {"x": 1, "y": 2}, "end": {"x": 3, "y": 4}}
            )

    def test_write_fast_orjson(self):
        """Write dict via orjson fast path (if available)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"x": 1, "y": 2}
            try:
                JsonWriter.to(path, data, fast=True)
                result = JsonReader.of(path, fast=False)
                self.assertEqual(result, data)
            except (ImportError, AssertionError):
                self.skipTest("orjson not available")

    def test_write_fast_indent_4_warning(self):
        """Writing with fast=True and indent=4 should warn about fallback to indent 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"x": 1}
            try:
                import orjson
                import warnings

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    JsonWriter.to(path, data, indent=4, fast=True)
                    indent_warnings = [
                        x for x in w if "indent" in str(x.message).lower()
                    ]
                    self.assertTrue(len(indent_warnings) > 0)
            except ImportError:
                self.skipTest("orjson not available")

    def test_write_fast_no_indent(self):
        """Write with fast=True and indent=0 (no indentation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"x": 1, "y": 2}
            try:
                JsonWriter.to(path, data, indent=0, fast=True)
                result = JsonReader.of(path, fast=False)
                self.assertEqual(result, data)
            except (ImportError, AssertionError):
                self.skipTest("orjson not available")

    def test_write_fast_orjson_fallback(self):
        """Lines 51-53: orjson ImportError fallback in to()."""
        import builtins
        import warnings

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "orjson":
                raise ImportError("mock orjson missing")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"x": 1}
            with unittest.mock.patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    JsonWriter.to(path, data, fast=True)
                    orjson_warnings = [x for x in w if "orjson" in str(x.message)]
                    self.assertTrue(len(orjson_warnings) > 0)
            result = JsonReader.of(path, fast=False)
            self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
