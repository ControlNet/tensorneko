import os
import unittest
from typing import List

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
        data = read.json.of(os.path.join("test", "resource", "test_read_json", "test_read_json_as_dict.json"))
        self.assertEqual(data, {"x": 1, "y": 2})

    def test_read_json_as_list(self):
        data = read.json.of(os.path.join("test", "resource", "test_read_json", "test_read_json_as_list.json"))
        self.assertEqual(data, [{"x": 1, "y": 2}, {"x": 3, "y": 2}])

    def test_read_json_as_obj(self):
        data: Point = read.json.of(
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_dict.json"),
            clazz=Point
        )
        self.assertEqual(str(data), "Point(x=1, y=2)")
        self.assertEqual(data.x, 1)
        self.assertEqual(data.y, 2)

    def test_read_json_as_list_of_objs(self):
        data: List[Point] = read.json.of(
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_list.json"),
            clazz=List[Point]
        )
        self.assertEqual(str(data), "[Point(x=1, y=2), Point(x=3, y=2)]")
        self.assertEqual(data[0].x, 1)
        self.assertEqual(data[0].y, 2)
        self.assertEqual(data[1].x, 3)
        self.assertEqual(data[1].y, 2)

    def test_read_json_as_obj_in_obj(self):
        # test obj in obj with method
        line: Line = read.json.of(
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_obj_in_obj.json"),
            clazz=Line
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
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_list_in_obj.json"),
            clazz=Points
        )

        self.assertEqual(str(points), "Points(instances=[Point(x=1, y=2), Point(x=3, y=2)])")
        self.assertEqual(points.instances[0].x, 1)
        self.assertEqual(points.instances[0].y, 2)
        self.assertEqual(points.instances[1].x, 3)
        self.assertEqual(points.instances[1].y, 2)

    def test_read_json_as_obj_in_obj_in_obj(self):
        # test obj in obj in obj
        triangle: Triangle = read.json.of(
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_obj_in_obj_in_obj.json"),
            clazz=Triangle
        )

        self.assertEqual(str(triangle), "Triangle(line1=Line(start=Point(x=1, y=1), end=Point(x=4, y=5)), "
                                        "line2=Line(start=Point(x=4, y=5), end=Point(x=2, y=5)), "
                                        "line3=Line(start=Point(x=2, y=5), end=Point(x=1, y=1)))")

        self.assertEqual(triangle.line1.start.x, 1)
        self.assertEqual(triangle.line1.end.x, 4)
        self.assertEqual(triangle.line2.start.x, 4)
        self.assertEqual(triangle.line2.end.x, 2)
        self.assertEqual(triangle.line3.start.x, 2)
        self.assertEqual(triangle.line3.end.x, 1)

    def test_read_json_as_list_in_list(self):
        # test list in list
        matrix: Matrix = read.json.of(
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_list_in_list.json"),
            clazz=Matrix
        )
        self.assertEqual(matrix.values[0][0], 1)
        self.assertEqual(matrix.values[1][2], 6)
        self.assertEqual(matrix.values[2][1], 8)

    def test_read_json_as_obj_in_list_in_list(self):
        # test obj in list
        matrix_info: MatrixInfo = read.json.of(
            os.path.join("test", "resource", "test_read_json", "test_read_json_as_obj_in_list_in_list.json"),
            clazz=MatrixInfo
        )

        self.assertEqual(matrix_info.values[0][0].name, "A")
        self.assertEqual(matrix_info.values[1][2].name, "F")
        self.assertEqual(matrix_info.values[2][1].name, "H")
