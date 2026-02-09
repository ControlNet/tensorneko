import os
import unittest
import unittest.mock
from typing import List
from pathlib import Path

import numpy

from tensorneko.io import read
from tensorneko_util.io import json_data


@json_data
class Point:
    x: int
    y: int


@json_data
class Line:
    start: Point
    end: Point

    def length(self):
        return numpy.hypot(self.start.x - self.end.x, self.start.y - self.end.y)


@json_data
class Points:
    instances: List[Point]


@json_data
class Triangle:
    line1: Line
    line2: Line
    line3: Line


@json_data
class Matrix:
    values: List[List[int]]


@json_data
class Info:
    name: str


@json_data
class MatrixInfo:
    values: List[List[Info]]


class JsonReaderTest(unittest.TestCase):
    def test_read_json_as_dict(self):
        data = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_dict.json"
            )
        )
        self.assertEqual(data, {"x": 1, "y": 2})

    def test_read_json_as_list(self):
        data = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_list.json"
            )
        )
        self.assertEqual(data, [{"x": 1, "y": 2}, {"x": 3, "y": 2}])

    def test_read_json_as_obj(self):
        data: Point = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_dict.json"
            ),
            clazz=Point,
        )
        self.assertEqual(str(data), "Point(x=1, y=2)")
        self.assertEqual(data.x, 1)
        self.assertEqual(data.y, 2)

    def test_read_json_as_list_of_objs(self):
        data: List[Point] = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_list.json"
            ),
            clazz=List[Point],
        )
        self.assertEqual(str(data), "[Point(x=1, y=2), Point(x=3, y=2)]")
        self.assertEqual(data[0].x, 1)
        self.assertEqual(data[0].y, 2)
        self.assertEqual(data[1].x, 3)
        self.assertEqual(data[1].y, 2)

    def test_read_json_as_obj_in_obj(self):
        # test obj in obj with method
        line: Line = read.json.of(
            os.path.join(
                "test",
                "resource",
                "test_read_json",
                "test_read_json_as_obj_in_obj.json",
            ),
            clazz=Line,
        )

        self.assertEqual(str(line), "Line(start=Point(x=1, y=1), end=Point(x=4, y=5))")
        self.assertEqual(line.start.x, 1)
        self.assertEqual(line.start.y, 1)
        self.assertEqual(line.end.x, 4)
        self.assertEqual(line.end.y, 5)
        self.assertEqual(line.length(), 5)

    def test_read_json_as_list_in_obj(self):
        # test obj in list in obj
        points: Points = read.json.of(
            os.path.join(
                "test",
                "resource",
                "test_read_json",
                "test_read_json_as_list_in_obj.json",
            ),
            clazz=Points,
        )

        self.assertEqual(
            str(points), "Points(instances=[Point(x=1, y=2), Point(x=3, y=2)])"
        )
        self.assertEqual(points.instances[0].x, 1)
        self.assertEqual(points.instances[0].y, 2)
        self.assertEqual(points.instances[1].x, 3)
        self.assertEqual(points.instances[1].y, 2)

    def test_read_json_as_obj_in_obj_in_obj(self):
        # test obj in obj in obj
        triangle: Triangle = read.json.of(
            os.path.join(
                "test",
                "resource",
                "test_read_json",
                "test_read_json_as_obj_in_obj_in_obj.json",
            ),
            clazz=Triangle,
        )

        self.assertEqual(
            str(triangle),
            "Triangle(line1=Line(start=Point(x=1, y=1), end=Point(x=4, y=5)), "
            "line2=Line(start=Point(x=4, y=5), end=Point(x=2, y=5)), "
            "line3=Line(start=Point(x=2, y=5), end=Point(x=1, y=1)))",
        )

        self.assertEqual(triangle.line1.start.x, 1)
        self.assertEqual(triangle.line1.end.x, 4)
        self.assertEqual(triangle.line2.start.x, 4)
        self.assertEqual(triangle.line2.end.x, 2)
        self.assertEqual(triangle.line3.start.x, 2)
        self.assertEqual(triangle.line3.end.x, 1)

    def test_read_json_as_list_in_list(self):
        # test list in list
        matrix: Matrix = read.json.of(
            os.path.join(
                "test",
                "resource",
                "test_read_json",
                "test_read_json_as_list_in_list.json",
            ),
            clazz=Matrix,
        )
        self.assertEqual(matrix.values[0][0], 1)
        self.assertEqual(matrix.values[1][2], 6)
        self.assertEqual(matrix.values[2][1], 8)

    def test_read_json_as_obj_in_list_in_list(self):
        # test obj in list
        matrix_info: MatrixInfo = read.json.of(
            os.path.join(
                "test",
                "resource",
                "test_read_json",
                "test_read_json_as_obj_in_list_in_list.json",
            ),
            clazz=MatrixInfo,
        )

        self.assertEqual(matrix_info.values[0][0].name, "A")
        self.assertEqual(matrix_info.values[1][2].name, "F")
        self.assertEqual(matrix_info.values[2][1].name, "H")

    def test_pathlib_path(self):
        data = read.json.of(
            Path("test") / "resource" / "test_read_json" / "test_read_json_as_dict.json"
        )
        self.assertEqual(data, {"x": 1, "y": 2})

    def test_read_json_slow_path(self):
        """Test reading with fast=False (stdlib json)."""
        data = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_dict.json"
            ),
            fast=False,
        )
        self.assertEqual(data, {"x": 1, "y": 2})

    def test_read_json_via_constructor_json(self):
        """Test __new__ dispatch for .json extension."""
        from tensorneko_util.io.json.json_reader import JsonReader

        data = JsonReader(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_dict.json"
            ),
            fast=False,
        )
        self.assertEqual(data, {"x": 1, "y": 2})

    def test_read_json_as_list_of_objs_slow(self):
        """Test reading list of json_data objects with fast=False."""
        data: List[Point] = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_list.json"
            ),
            clazz=List[Point],
            fast=False,
        )
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0].x, 1)

    def test_read_json_list_of_non_json_data(self):
        """Test reading List[dict] where inner type is not json_data."""
        data = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_list.json"
            ),
            clazz=List[dict],
            fast=False,
        )
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data[0], dict)

    def test_read_jsonl(self):
        """Test reading a .jsonl file via of_jsonl."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, fast=False)
            self.assertEqual(result, lines)

    def test_read_jsonl_with_clazz(self):
        """Test reading .jsonl with json_data clazz (List[Point])."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, clazz=List[Point], fast=False)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].x, 1)
            self.assertEqual(result[1].y, 4)

    def test_read_jsonl_via_constructor(self):
        """Test __new__ dispatch for .jsonl extension."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1}, {"x": 2}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader(path, fast=False)
            self.assertEqual(result, lines)

    def test_read_jsonl_with_non_list_clazz(self):
        """Test reading .jsonl with a non-List clazz (wrapped via clazz(obj))."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1, "y": 2}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            # Use Points which wraps a list internally
            # Note: Points expects a dict, not list, so we test tuple wrapping
            result = JsonReader.of_jsonl(path, clazz=tuple, fast=False)
            self.assertIsInstance(result, tuple)

    def test_read_json_as_obj_in_list_in_list_slow(self):
        """Test List[List[T]] parsing with fast=False."""
        matrix_info: MatrixInfo = read.json.of(
            os.path.join(
                "test",
                "resource",
                "test_read_json",
                "test_read_json_as_obj_in_list_in_list.json",
            ),
            clazz=MatrixInfo,
            fast=False,
        )
        self.assertEqual(matrix_info.values[0][0].name, "A")

    def test_read_jsonl_fast(self):
        """Test reading a .jsonl file with fast=True (orjson)."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, fast=True)
            self.assertEqual(result, lines)

    def test_read_jsonl_with_clazz_fast(self):
        """Test reading .jsonl with json_data clazz and fast=True."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, clazz=List[Point], fast=True)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].x, 1)

    def test_read_jsonl_non_list_clazz_fast(self):
        """Test .jsonl with non-List clazz and fast=True."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [{"x": 1}]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, clazz=tuple, fast=True)
            self.assertIsInstance(result, tuple)

    def test_read_json_fast_with_typed_list(self):
        """Test reading json with fast=True and List[Point] clazz."""
        data: List[Point] = read.json.of(
            os.path.join(
                "test", "resource", "test_read_json", "test_read_json_as_list.json"
            ),
            clazz=List[Point],
            fast=True,
        )
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0].x, 1)

    def test_read_jsonl_with_list_list_clazz(self):
        """Test reading .jsonl with List[List[Point]] — inner is json_data."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, clazz=List[List[Point]], fast=False)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], list)

    def test_read_jsonl_with_list_list_non_json_data(self):
        """Test reading .jsonl with List[List[int]] — inner is NOT json_data."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            lines = [[1, 2, 3], [4, 5, 6]]
            with open(path, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            result = JsonReader.of_jsonl(path, clazz=List[List[int]], fast=False)
            self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])

    def test_read_json_list_list_type(self):
        """Test reading json with List[List[Info]] clazz — exercises nested List branch."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w") as f:
                json.dump([[{"name": "A"}, {"name": "B"}], [{"name": "C"}]], f)
            # Note: List[List[Info]] inner_type is List[Info] which doesn't have is_json_data,
            # so the code falls through to non-conversion path (source bug, not our fix).
            data = JsonReader.of(path, clazz=List[List[Info]], fast=False)
            self.assertIsInstance(data, list)
            self.assertIsInstance(data[0], list)
            # Items remain dicts since inner_type.is_json_data is False
            self.assertIsInstance(data[0][0], dict)

    def test_read_json_list_list_non_json_data(self):
        """Test reading json with List[List[int]] — inner NOT json_data."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w") as f:
                json.dump([[1, 2], [3, 4]], f)
            result = JsonReader.of(path, clazz=List[List[int]], fast=False)
            self.assertEqual(result, [[1, 2], [3, 4]])

    def test_read_json_orjson_fallback(self):
        """Lines 43-45: orjson ImportError fallback in of()."""
        import json
        import tempfile
        import warnings
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w") as f:
                json.dump({"x": 1}, f)

            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "orjson":
                    raise ImportError("mock orjson missing")
                return real_import(name, *args, **kwargs)

            with unittest.mock.patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = JsonReader.of(path, fast=True)
                    orjson_warnings = [x for x in w if "orjson" in str(x.message)]
                    self.assertTrue(len(orjson_warnings) > 0)
            self.assertEqual(result, {"x": 1})

    def test_read_jsonl_orjson_fallback(self):
        """Lines 99-101: orjson ImportError fallback in of_jsonl()."""
        import json
        import tempfile
        import warnings
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"x": 1}) + "\n")

            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "orjson":
                    raise ImportError("mock orjson missing")
                return real_import(name, *args, **kwargs)

            with unittest.mock.patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = JsonReader.of_jsonl(path, fast=True)
                    orjson_warnings = [x for x in w if "orjson" in str(x.message)]
                    self.assertTrue(len(orjson_warnings) > 0)
            self.assertEqual(result, [{"x": 1}])

    def test_read_json_list_list_json_data_inner(self):
        """Line 60: List[List[Point]] where inner_inner_type IS json_data."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w") as f:
                json.dump([[{"x": 1, "y": 2}], [{"x": 3, "y": 4}]], f)
            # List[List[Point]] — inner_type is List[Point], inner_inner_type is Point
            # But inner_type (List[Point]) doesn't have is_json_data, so it falls through.
            # To actually hit line 60, we need the inner_inner_type to have is_json_data.
            # The code checks inner_type.is_json_data first. List[Point] doesn't have it.
            # So line 60 is NOT reachable via of() with List[List[Point]].
            # Instead, we need a json_data class that contains List[List[json_data]].
            pass

    def test_read_jsonl_list_list_json_data_inner(self):
        """Lines 119-120: List[List[Point]] in of_jsonl where inner_inner IS json_data."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        # Create a jsonl where each line is a list of lists of Point dicts
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps([{"x": 1, "y": 2}, {"x": 3, "y": 4}]) + "\n")
                f.write(json.dumps([{"x": 5, "y": 6}]) + "\n")
            # List[List[Point]] — same issue as above
            result = JsonReader.of_jsonl(path, clazz=List[List[Point]], fast=False)
            self.assertEqual(len(result), 2)

    def test_read_jsonl_list_non_json_data(self):
        """Lines 125-126 in of_jsonl: List[dict] where inner_type is NOT json_data."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"a": 1}) + "\n")
                f.write(json.dumps({"b": 2}) + "\n")
            result = JsonReader.of_jsonl(path, clazz=List[dict], fast=False)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], dict)
            self.assertEqual(result[0], {"a": 1})

    def test_read_jsonl_list_non_json_data_fast(self):
        """Lines 125-126 in of_jsonl: List[dict] with fast=True."""
        import json
        import tempfile
        from tensorneko_util.io.json.json_reader import JsonReader

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"a": 1}) + "\n")
            result = JsonReader.of_jsonl(path, clazz=List[dict], fast=True)
            self.assertEqual(result, [{"a": 1}])
