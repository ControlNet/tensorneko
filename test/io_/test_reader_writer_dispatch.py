import unittest
import tempfile
from unittest.mock import patch
from collections import OrderedDict

import torch

from tensorneko.io import Reader, Writer


class TestReaderDispatch(unittest.TestCase):
    """Test Reader.__call__ dispatch logic."""

    def test_dispatch_pt_to_weight_reader(self):
        """Test that .pt files are dispatched to WeightReader."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            # Create test weights
            weights = OrderedDict([("tensor", torch.randn(2, 2))])
            torch.save(weights, tmp.name)

            # Use Reader to load
            reader = Reader()
            loaded = reader(tmp.name)

            # Verify it was loaded correctly
            self.assertIn("tensor", loaded)
            self.assertTrue(torch.allclose(loaded["tensor"], weights["tensor"]))

    def test_dispatch_pth_to_weight_reader(self):
        """Test that .pth files are dispatched to WeightReader."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            weights = OrderedDict([("param", torch.randn(3, 3))])
            torch.save(weights, tmp.name)

            reader = Reader()
            loaded = reader(tmp.name)

            self.assertIn("param", loaded)
            self.assertTrue(torch.allclose(loaded["param"], weights["param"]))

    def test_dispatch_ckpt_to_weight_reader(self):
        """Test that .ckpt files are dispatched to WeightReader."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            state_dict = OrderedDict([("layer.weight", torch.randn(4, 4))])
            # .ckpt files contain state_dict nested
            ckpt_data = {"state_dict": state_dict}
            torch.save(ckpt_data, tmp.name)

            reader = Reader()
            loaded = reader(tmp.name)

            # WeightReader.of_ckpt extracts the state_dict
            self.assertIn("layer.weight", loaded)
            self.assertTrue(
                torch.allclose(loaded["layer.weight"], state_dict["layer.weight"])
            )

    def test_dispatch_safetensors_to_weight_reader(self):
        """Test that .safetensors files are dispatched to WeightReader."""
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors not installed")

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            weights = OrderedDict([("weight", torch.randn(5, 5))])
            safetensors.torch.save_file(weights, tmp.name)

            reader = Reader()
            loaded = reader(tmp.name)

            self.assertIn("weight", loaded)
            self.assertTrue(torch.allclose(loaded["weight"], weights["weight"]))

    def test_dispatch_txt_to_base_reader(self):
        """Test that .txt files are dispatched to BaseReader.__call__."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
            test_content = "Hello, World!"
            tmp.write(test_content)
            tmp.flush()

            reader = Reader()
            loaded = reader(tmp.name)

            # BaseReader for txt should return the text content
            self.assertEqual(loaded, test_content)

    def test_mesh_property_import_error(self):
        """Test that accessing mesh property raises ImportError when pytorch3d is not available."""
        reader = Reader()

        # Mock pytorch3d_available to False
        with patch("tensorneko.io.reader.pytorch3d_available", False):
            with self.assertRaises(ImportError) as context:
                _ = reader.mesh

            self.assertIn("pytorch3d", str(context.exception))


class TestWriterDispatch(unittest.TestCase):
    """Test Writer.__call__ dispatch logic."""

    def test_dispatch_pt_to_weight_writer(self):
        """Test that .pt files are dispatched to WeightWriter."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            weights = OrderedDict([("tensor", torch.randn(2, 2))])

            writer = Writer()
            writer(tmp.name, weights)

            # Verify file was written correctly
            loaded = torch.load(tmp.name)
            self.assertIn("tensor", loaded)
            self.assertTrue(torch.allclose(loaded["tensor"], weights["tensor"]))

    def test_dispatch_pth_to_weight_writer(self):
        """Test that .pth files are dispatched to WeightWriter."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            weights = OrderedDict([("param", torch.randn(3, 3))])

            writer = Writer()
            writer(tmp.name, weights)

            loaded = torch.load(tmp.name)
            self.assertTrue(torch.allclose(loaded["param"], weights["param"]))

    def test_dispatch_ckpt_to_weight_writer_not_supported(self):
        """Test that .ckpt files are dispatched to WeightWriter but raise error."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            weights = OrderedDict([("layer.weight", torch.randn(4, 4))])

            writer = Writer()
            # .ckpt is routed to WeightWriter but not supported by WeightWriter.to()
            with self.assertRaises(ValueError) as context:
                writer(tmp.name, weights)

            self.assertIn("Unknown file type", str(context.exception))

    def test_dispatch_safetensors_to_weight_writer(self):
        """Test that .safetensors files are dispatched to WeightWriter."""
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors not installed")

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            weights = OrderedDict([("weight", torch.randn(5, 5))])

            writer = Writer()
            writer(tmp.name, weights)

            # Verify with safetensors
            loaded = safetensors.torch.load_file(tmp.name)
            self.assertIn("weight", loaded)
            self.assertTrue(torch.allclose(loaded["weight"], weights["weight"]))

    def test_dispatch_txt_to_base_writer(self):
        """Test that .txt files are dispatched to BaseWriter.__call__."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
            test_content = "Test message"

            writer = Writer()
            writer(tmp.name, test_content)

            # Read back using BaseWriter
            with open(tmp.name, "r") as f:
                loaded = f.read()

            self.assertEqual(loaded, test_content)

    def test_mesh_property_import_error(self):
        """Test that accessing mesh property raises ImportError when pytorch3d is not available."""
        writer = Writer()

        # Mock pytorch3d_available to False
        with patch("tensorneko.io.writer.pytorch3d_available", False):
            with self.assertRaises(ImportError) as context:
                _ = writer.mesh

            self.assertIn("pytorch3d", str(context.exception))


if __name__ == "__main__":
    unittest.main()
