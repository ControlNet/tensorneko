import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from tensorneko_util.io.hdf5.hdf5_reader import Hdf5Reader
from tensorneko_util.io.hdf5.hdf5_writer import Hdf5Writer


class Hdf5ReaderTest(unittest.TestCase):
    def test_read_hdf5_file(self):
        """Create an HDF5 file with h5py, read back via Hdf5Reader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.h5")
            arr = np.array([1.0, 2.0, 3.0])
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=arr)

            result = Hdf5Reader.of(path)
            np.testing.assert_array_equal(result["data"][:], arr)
            result.close()

    def test_read_via_constructor(self):
        """Test __new__ alias for Hdf5Reader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.h5")
            arr = np.arange(10)
            with h5py.File(path, "w") as f:
                f.create_dataset("values", data=arr)

            result = Hdf5Reader(path)
            np.testing.assert_array_equal(result["values"][:], arr)
            result.close()

    def test_read_multiple_datasets(self):
        """Read an HDF5 file with multiple datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi.h5")
            a = np.array([1, 2, 3])
            b = np.array([[4.0, 5.0], [6.0, 7.0]])
            with h5py.File(path, "w") as f:
                f.create_dataset("a", data=a)
                f.create_dataset("b", data=b)

            result = Hdf5Reader.of(path)
            np.testing.assert_array_equal(result["a"][:], a)
            np.testing.assert_array_equal(result["b"][:], b)
            self.assertEqual(len(result.keys()), 2)
            result.close()

    def test_read_pathlib_path(self):
        """Test Hdf5Reader with pathlib.Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.h5"
            arr = np.array([10, 20])
            with h5py.File(str(path), "w") as f:
                f.create_dataset("x", data=arr)

            result = Hdf5Reader.of(path)
            np.testing.assert_array_equal(result["x"][:], arr)
            result.close()

    def test_read_with_groups(self):
        """Read HDF5 file with groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grouped.h5")
            with h5py.File(path, "w") as f:
                grp = f.create_group("group1")
                grp.create_dataset("data", data=np.array([1, 2, 3]))

            result = Hdf5Reader.of(path)
            np.testing.assert_array_equal(
                result["group1"]["data"][:], np.array([1, 2, 3])
            )
            result.close()


class Hdf5WriterTest(unittest.TestCase):
    def test_writer_to_not_implemented(self):
        """Hdf5Writer.to should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            Hdf5Writer.to("dummy.h5")

    def test_writer_constructor_not_implemented(self):
        """Hdf5Writer() should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            Hdf5Writer("dummy.h5")


if __name__ == "__main__":
    unittest.main()
