import unittest
import tempfile
import os
from pathlib import Path
from collections import OrderedDict

import torch

from tensorneko.io.weight import WeightReader, WeightWriter


class TestWeightReader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_save_and_load_pt(self):
        """Test saving and loading .pt files."""
        # Create sample weights
        weights = OrderedDict(
            [
                ("layer1.weight", torch.randn(10, 5)),
                ("layer1.bias", torch.randn(10)),
                ("layer2.weight", torch.randn(5, 10)),
            ]
        )

        # Save weights
        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter.to_pt(save_path, weights)

        # Load weights
        loaded = WeightReader.of_pt(save_path)

        # Verify
        self.assertEqual(len(loaded), len(weights))
        for key in weights:
            self.assertIn(key, loaded)
            self.assertTrue(torch.allclose(loaded[key], weights[key]))

    def test_save_and_load_pth(self):
        """Test saving and loading .pth files."""
        weights = OrderedDict(
            [
                ("weight", torch.ones(3, 3)),
                ("bias", torch.zeros(3)),
            ]
        )

        # Save as .pth
        save_path = os.path.join(self.temp_path, "model.pth")
        WeightWriter.to_pt(save_path, weights)

        # Load from .pth
        loaded = WeightReader.of_pt(save_path)

        self.assertEqual(len(loaded), 2)
        self.assertTrue(torch.allclose(loaded["weight"], weights["weight"]))
        self.assertTrue(torch.allclose(loaded["bias"], weights["bias"]))

    def test_writer_to_generic(self):
        """Test generic WeightWriter.to() method."""
        weights = OrderedDict(
            [
                ("param1", torch.randn(2, 2)),
                ("param2", torch.randn(4)),
            ]
        )

        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter.to(save_path, weights)

        # Verify file was created
        self.assertTrue(os.path.exists(save_path))

    def test_reader_generic(self):
        """Test generic WeightReader.of() method."""
        weights = OrderedDict(
            [
                ("linear.weight", torch.randn(5, 10)),
            ]
        )

        # Save and load using generic method
        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        self.assertTrue(
            torch.allclose(loaded["linear.weight"], weights["linear.weight"])
        )

    def test_reader_by_extension_pt(self):
        """Test that WeightReader.of auto-detects .pt file."""
        weights = OrderedDict([("tensor", torch.tensor([1, 2, 3]))])

        save_path = os.path.join(self.temp_path, "weights.pt")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        self.assertTrue(torch.allclose(loaded["tensor"], weights["tensor"]))

    def test_reader_by_extension_pth(self):
        """Test that WeightReader.of auto-detects .pth file."""
        weights = OrderedDict([("tensor", torch.tensor([1.0, 2.0]))])

        save_path = os.path.join(self.temp_path, "weights.pth")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        self.assertTrue(torch.allclose(loaded["tensor"], weights["tensor"]))

    def test_reader_invalid_extension(self):
        """Test that WeightReader raises error for unsupported formats."""
        save_path = os.path.join(self.temp_path, "model.invalid")

        with self.assertRaises(ValueError):
            WeightReader.of(save_path)

    def test_writer_invalid_extension(self):
        """Test that WeightWriter raises error for unsupported formats."""
        weights = OrderedDict([("tensor", torch.randn(2, 2))])
        save_path = os.path.join(self.temp_path, "model.invalid")

        with self.assertRaises(ValueError):
            WeightWriter.to(save_path, weights)

    def test_round_trip_with_different_dtypes(self):
        """Test round-trip with different tensor dtypes."""
        weights = OrderedDict(
            [
                ("float32", torch.randn(2, 2).float()),
                ("float64", torch.randn(2, 2).double()),
                ("int32", torch.randint(0, 10, (2, 2)).int()),
            ]
        )

        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        for key in weights:
            self.assertEqual(loaded[key].dtype, weights[key].dtype)
            self.assertTrue(torch.allclose(loaded[key].float(), weights[key].float()))

    def test_large_state_dict(self):
        """Test with large state dictionary."""
        weights = OrderedDict()
        for i in range(100):
            weights[f"layer_{i}.weight"] = torch.randn(64, 64)
            weights[f"layer_{i}.bias"] = torch.randn(64)

        save_path = os.path.join(self.temp_path, "large_model.pt")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        self.assertEqual(len(loaded), len(weights))

    def test_weight_reader_with_pathlib_path(self):
        """Test WeightReader with pathlib.Path objects."""
        weights = OrderedDict([("param", torch.randn(2, 2))])

        save_path = Path(self.temp_path) / "model.pt"
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        self.assertTrue(torch.allclose(loaded["param"], weights["param"]))

    def test_weight_writer_with_pathlib_path(self):
        """Test WeightWriter with pathlib.Path objects."""
        weights = OrderedDict([("param", torch.randn(3, 3))])

        save_path = Path(self.temp_path) / "model.pt"
        WeightWriter.to(save_path, weights)

        self.assertTrue(save_path.exists())

    def test_reader_callable(self):
        """Test that WeightReader can be called directly."""
        weights = OrderedDict([("tensor", torch.tensor([1, 2, 3]))])

        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter.to(save_path, weights)

        # Call WeightReader directly
        loaded = WeightReader(save_path)

        self.assertTrue(torch.allclose(loaded["tensor"], weights["tensor"]))

    def test_writer_callable(self):
        """Test that WeightWriter can be called directly."""
        weights = OrderedDict([("tensor", torch.tensor([1.0, 2.0]))])

        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter(save_path, weights)

        self.assertTrue(os.path.exists(save_path))

    def test_empty_state_dict(self):
        """Test round-trip with empty state dict."""
        weights = OrderedDict()

        save_path = os.path.join(self.temp_path, "empty.pt")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        self.assertEqual(len(loaded), 0)

    def test_map_location_parameter(self):
        """Test WeightReader with map_location parameter."""
        weights = OrderedDict([("tensor", torch.randn(2, 2))])

        save_path = os.path.join(self.temp_path, "model.pt")
        WeightWriter.to(save_path, weights)

        # Load with explicit map_location
        loaded = WeightReader.of(save_path, map_location="cpu")

        self.assertIn("tensor", loaded)

    def test_nested_keys(self):
        """Test with nested module names (dots in keys)."""
        weights = OrderedDict(
            [
                ("module.layer.weight", torch.randn(10, 5)),
                ("module.layer.bias", torch.randn(10)),
                ("module.head.weight", torch.randn(3, 10)),
            ]
        )

        save_path = os.path.join(self.temp_path, "nested.pt")
        WeightWriter.to(save_path, weights)
        loaded = WeightReader.of(save_path)

        for key in weights:
            self.assertIn(key, loaded)
            self.assertTrue(torch.allclose(loaded[key], weights[key]))

    def test_of_ckpt(self):
        """Test WeightReader.of_ckpt() method."""
        weights = OrderedDict(
            [
                ("model.weight", torch.randn(8, 8)),
                ("model.bias", torch.randn(8)),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            # Save state_dict as .ckpt format (wrapped in dict)
            ckpt_data = {"state_dict": weights}
            torch.save(ckpt_data, tmp.name)

            # Read back with of_ckpt
            loaded = WeightReader.of_ckpt(tmp.name)

            # Verify
            self.assertEqual(len(loaded), len(weights))
            for key in weights:
                self.assertIn(key, loaded)
                self.assertTrue(torch.allclose(loaded[key], weights[key]))

    def test_of_safetensors(self):
        """Test WeightReader.of_safetensors() method."""
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors not installed")

        weights = OrderedDict(
            [
                ("conv.weight", torch.randn(16, 3, 3, 3)),
                ("conv.bias", torch.randn(16)),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            # Save using safetensors
            safetensors.torch.save_file(weights, tmp.name)

            # Read back with of_safetensors
            loaded = WeightReader.of_safetensors(tmp.name)

            # Verify
            self.assertEqual(len(loaded), len(weights))
            for key in weights:
                self.assertIn(key, loaded)
                self.assertTrue(torch.allclose(loaded[key], weights[key]))

    def test_to_safetensors(self):
        """Test WeightWriter.to_safetensors() method."""
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors not installed")

        weights = OrderedDict(
            [
                ("layer1.weight", torch.randn(32, 64)),
                ("layer1.bias", torch.randn(32)),
                ("layer2.weight", torch.randn(10, 32)),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            # Write using to_safetensors
            WeightWriter.to_safetensors(tmp.name, weights)

            # Read back
            loaded = safetensors.torch.load_file(tmp.name)

            # Verify
            self.assertEqual(len(loaded), len(weights))
            for key in weights:
                self.assertIn(key, loaded)
                self.assertTrue(torch.allclose(loaded[key], weights[key]))

    def test_of_with_ckpt(self):
        """Test WeightReader.of() auto-detects .ckpt files."""
        weights = OrderedDict([("param", torch.randn(5, 5))])

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            # Save in .ckpt format
            ckpt_data = {"state_dict": weights}
            torch.save(ckpt_data, tmp.name)

            # Read with generic .of() method
            loaded = WeightReader.of(tmp.name)

            # Verify it extracts the state_dict
            self.assertIn("param", loaded)
            self.assertTrue(torch.allclose(loaded["param"], weights["param"]))

    def test_to_with_safetensors(self):
        """Test WeightWriter.to() auto-detects .safetensors files."""
        try:
            import safetensors.torch
        except ImportError:
            self.skipTest("safetensors not installed")

        weights = OrderedDict(
            [
                ("encoder.weight", torch.randn(128, 256)),
                ("decoder.weight", torch.randn(256, 128)),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            # Write using generic .to() method
            WeightWriter.to(tmp.name, weights)

            # Read back
            loaded = safetensors.torch.load_file(tmp.name)

            # Verify
            self.assertEqual(len(loaded), len(weights))
            for key in weights:
                self.assertIn(key, loaded)
                self.assertTrue(torch.allclose(loaded[key], weights[key]))


if __name__ == "__main__":
    unittest.main()
