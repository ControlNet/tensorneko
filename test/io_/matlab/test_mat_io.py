import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorneko_util.io.matlab.mat_reader import MatReader
from tensorneko_util.io.matlab.mat_writer import MatWriter


class TestMatWriter(unittest.TestCase):
    def test_write_with_str_path(self):
        data = {"array": np.array([1, 2, 3]), "scalar": np.array(42)}
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            MatWriter.to(path, data)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_write_with_pathlib_path(self):
        data = {"array": np.array([4, 5, 6])}
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = Path(f.name)
        try:
            MatWriter.to(path, data)
            self.assertTrue(path.exists())
        finally:
            path.unlink()

    def test_write_via_constructor(self):
        data = {"x": np.array([10, 20])}
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            MatWriter(path, data)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


class TestMatReader(unittest.TestCase):
    def _write_temp_mat(self, data):
        f = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
        path = f.name
        f.close()
        MatWriter.to(path, data)
        return path

    def test_read_with_str_path(self):
        data = {"array": np.array([1, 2, 3]), "scalar": np.array(42)}
        path = self._write_temp_mat(data)
        try:
            result = MatReader.of(path)
            np.testing.assert_array_equal(result["array"].squeeze(), data["array"])
            np.testing.assert_array_equal(result["scalar"].squeeze(), data["scalar"])
        finally:
            os.unlink(path)

    def test_read_with_pathlib_path(self):
        data = {"values": np.array([7, 8, 9])}
        path = self._write_temp_mat(data)
        try:
            result = MatReader.of(Path(path))
            np.testing.assert_array_equal(result["values"].squeeze(), data["values"])
        finally:
            os.unlink(path)

    def test_read_via_constructor(self):
        data = {"x": np.array([10, 20])}
        path = self._write_temp_mat(data)
        try:
            result = MatReader(path)
            np.testing.assert_array_equal(result["x"].squeeze(), data["x"])
        finally:
            os.unlink(path)

    def test_roundtrip(self):
        data = {"array": np.array([1, 2, 3]), "scalar": np.array(42)}
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            MatWriter.to(path, data)
            result = MatReader.of(path)
            np.testing.assert_array_equal(result["array"].squeeze(), data["array"])
            np.testing.assert_array_equal(result["scalar"].squeeze(), data["scalar"])
        finally:
            os.unlink(path)

    def test_roundtrip_via_constructors(self):
        data = {"items": np.array([[1, 2], [3, 4]])}
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            MatWriter(path, data)
            result = MatReader(path)
            np.testing.assert_array_equal(result["items"], data["items"])
        finally:
            os.unlink(path)
