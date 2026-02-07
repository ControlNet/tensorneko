import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorneko_util.io.npy.npy_reader import NpyReader
from tensorneko_util.io.npy.npy_writer import NpyWriter


class NpyReaderWriterTest(unittest.TestCase):
    def test_roundtrip_npy(self):
        """Save and load a single numpy array via .npy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arr.npy")
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            NpyWriter.to(path, arr)
            loaded = NpyReader.of(path)
            np.testing.assert_array_equal(loaded, arr)

    def test_roundtrip_npy_via_constructor(self):
        """Test __new__ dispatch for .npy extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arr.npy")
            arr = np.arange(12).reshape(3, 4)
            NpyWriter(path, arr)
            loaded = NpyReader(path)
            np.testing.assert_array_equal(loaded, arr)

    def test_roundtrip_npz(self):
        """Save and load multiple arrays via .npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.npz")
            a = np.array([1, 2, 3])
            b = np.array([4.0, 5.0])
            NpyWriter.to_npz(path, a=a, b=b)
            loaded = NpyReader.of(path)
            np.testing.assert_array_equal(loaded["a"], a)
            np.testing.assert_array_equal(loaded["b"], b)

    def test_roundtrip_npz_compressed(self):
        """Save and load compressed .npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.npz")
            a = np.random.rand(100, 100)
            NpyWriter.to_npz(path, compressed=True, data=a)
            loaded = NpyReader.of(path)
            np.testing.assert_array_almost_equal(loaded["data"], a)

    def test_roundtrip_npz_via_constructor(self):
        """Test __new__ dispatch for .npz extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.npz")
            a = np.array([10, 20, 30])
            NpyWriter(path, x=a)
            loaded = NpyReader(path)
            np.testing.assert_array_equal(loaded["x"], a)

    def test_roundtrip_txt(self):
        """Save and load numpy array as text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.txt")
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            NpyWriter.to_txt(path, arr, delimiter=",")
            loaded = NpyReader.of_txt(path, delimiter=",")
            np.testing.assert_array_almost_equal(loaded, arr)

    def test_txt_via_constructor(self):
        """Test __new__ dispatch for .txt extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.txt")
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            NpyWriter(path, arr)
            loaded = NpyReader(path)
            np.testing.assert_array_almost_equal(loaded, arr)

    def test_pathlib_path(self):
        """Test with pathlib.Path input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "arr.npy"
            arr = np.array([1, 2, 3])
            NpyWriter.to(path, arr)
            loaded = NpyReader.of(path)
            np.testing.assert_array_equal(loaded, arr)

    def test_writer_unknown_extension_raises(self):
        """NpyWriter.__new__ should raise ValueError for unknown extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.xyz")
            with self.assertRaises(ValueError):
                NpyWriter(path, np.array([1]))

    def test_reader_fallback_extension(self):
        """NpyReader.__new__ falls back to np.load for unknown extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as .npy, rename to unknown ext, reader should still try np.load
            path_npy = os.path.join(tmpdir, "arr.npy")
            np.save(path_npy, np.array([1, 2, 3]))
            # Rename to .dat
            path_dat = os.path.join(tmpdir, "arr.dat")
            os.rename(path_npy, path_dat)
            loaded = NpyReader(path_dat)
            np.testing.assert_array_equal(loaded, np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
