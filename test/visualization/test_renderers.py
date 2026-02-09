import unittest
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from tensorneko.visualization.dataset_viewer.renderers import (
    render_media,
    render_metadata,
    get_media_content_type,
    _to_numpy,
)

PNG_HEADER = b"\x89PNG"


class TestRenderMedia(unittest.TestCase):
    """Tests for render_media (image rendering pipeline)."""

    def test_image_tensor_chw_float(self):
        """1. CHW float [0,1] tensor → PNG bytes with correct header."""
        tensor = torch.rand(3, 8, 8)  # float32, range [0,1]
        result = render_media(tensor, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        # Verify dimensions via PIL round-trip
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (8, 8))

    def test_image_tensor_chw_uint8(self):
        """2. CHW uint8 tensor → PNG bytes."""
        tensor = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        result = render_media(tensor, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (16, 16))

    def test_image_tensor_grayscale(self):
        """3. 1-channel grayscale (1, H, W) → PNG bytes."""
        tensor = torch.rand(1, 12, 12)
        result = render_media(tensor, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (12, 12))
        # Should be grayscale (mode 'L')
        self.assertEqual(img.mode, "L")

    def test_image_tensor_rgba(self):
        """3b. 4-channel RGBA tensor → PNG bytes."""
        tensor = torch.rand(4, 10, 10)
        result = render_media(tensor, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.mode, "RGBA")

    def test_image_pil(self):
        """4. PIL Image → PNG bytes."""
        pil_img = Image.new("RGB", (20, 20), color=(128, 64, 32))
        result = render_media(pil_img, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (20, 20))

    def test_image_numpy_chw(self):
        """5. numpy array CHW float → PNG bytes."""
        arr = np.random.rand(3, 8, 8).astype(np.float32)
        result = render_media(arr, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (8, 8))

    def test_image_numpy_grayscale(self):
        """5b. numpy array 1-channel grayscale → PNG bytes."""
        arr = np.random.rand(1, 8, 8).astype(np.float32)
        result = render_media(arr, "image")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result[:4] == PNG_HEADER)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.mode, "L")


class TestRenderMetadata(unittest.TestCase):
    """Tests for render_metadata (JSON-serializable summaries)."""

    def test_text_string(self):
        """6. Text string → metadata dict."""
        result = render_metadata("hello", "text")
        self.assertEqual(result, {"type": "text", "value": "hello"})

    def test_scalar_int(self):
        """7. Scalar int → metadata dict."""
        result = render_metadata(42, "scalar")
        self.assertEqual(result, {"type": "scalar", "value": 42})

    def test_scalar_float(self):
        """8. Scalar float → metadata dict."""
        result = render_metadata(3.14, "scalar")
        self.assertEqual(result, {"type": "scalar", "value": 3.14})

    def test_scalar_tensor(self):
        """8b. Scalar tensor → converts to Python float."""
        tensor = torch.tensor(2.718)
        result = render_metadata(tensor, "scalar")
        self.assertEqual(result["type"], "scalar")
        self.assertIsInstance(result["value"], float)
        self.assertAlmostEqual(result["value"], 2.718, places=3)

    def test_tensor_metadata(self):
        """9. Generic tensor → dict with shape, dtype, min, max, mean keys."""
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        result = render_metadata(tensor, "tensor")
        self.assertEqual(result["type"], "tensor")
        self.assertEqual(result["shape"], [3, 4])
        self.assertIn("float", result["dtype"])
        self.assertAlmostEqual(result["min"], 0.0)
        self.assertAlmostEqual(result["max"], 11.0)
        self.assertAlmostEqual(result["mean"], 5.5)
        # numel=12 < 100, so data should be included
        self.assertIsNotNone(result["data"])
        self.assertEqual(len(result["data"]), 3)  # 3 rows

    def test_tensor_metadata_large(self):
        """9b. Large tensor → data is None."""
        tensor = torch.randn(100, 100)  # numel=10000 > 100
        result = render_metadata(tensor, "tensor")
        self.assertEqual(result["type"], "tensor")
        self.assertEqual(result["shape"], [100, 100])
        self.assertIsNone(result["data"])

    def test_json_type(self):
        """10. JSON type (nested dict) → metadata dict."""
        payload = {"key": [1, 2, 3], "nested": {"a": True}}
        result = render_metadata(payload, "json")
        self.assertEqual(result, {"type": "json", "value": payload})

    def test_unknown_type(self):
        """11. Unknown type → str representation."""
        result = render_metadata(object(), "unknown_type")
        self.assertEqual(result["type"], "unknown")
        self.assertIsInstance(result["value"], str)

    def test_image_metadata(self):
        """Image field_type → just type marker."""
        result = render_metadata(torch.rand(3, 8, 8), "image")
        self.assertEqual(result, {"type": "image"})


class TestRenderMediaEdgeCases(unittest.TestCase):
    """Tests for render_media edge cases and error paths."""

    def test_render_media_non_image_raises(self):
        """render_media with non-image field_type → ValueError."""
        with self.assertRaises(ValueError) as ctx:
            render_media(torch.rand(10), "tensor")
        self.assertIn("tensor", str(ctx.exception))

    def test_to_numpy_unsupported_type_raises(self):
        """_to_numpy with unsupported type → TypeError."""
        with self.assertRaises(TypeError) as ctx:
            _to_numpy("not a tensor")
        self.assertIn("str", str(ctx.exception))


class TestRenderMetadataEdgeCases(unittest.TestCase):
    """Tests for render_metadata edge cases (numpy tensors, fallback)."""

    def test_numpy_tensor_small(self):
        """numpy array with 'tensor' type → metadata with shape/dtype/stats and data."""
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = render_metadata(arr, "tensor")
        self.assertEqual(result["type"], "tensor")
        self.assertEqual(result["shape"], [3, 4])
        self.assertIn("float64", result["dtype"])
        self.assertAlmostEqual(result["min"], 0.0)
        self.assertAlmostEqual(result["max"], 11.0)
        self.assertAlmostEqual(result["mean"], 5.5)
        # size=12 < 100, so data should be included
        self.assertIsNotNone(result["data"])
        self.assertEqual(len(result["data"]), 3)

    def test_numpy_tensor_large(self):
        """Large numpy array with 'tensor' type → data is None."""
        arr = np.random.randn(100, 100)
        result = render_metadata(arr, "tensor")
        self.assertEqual(result["type"], "tensor")
        self.assertEqual(result["shape"], [100, 100])
        self.assertIsNone(result["data"])

    def test_tensor_type_non_array_fallback(self):
        """Non-tensor/non-numpy value with 'tensor' type → fallback to unknown."""
        result = render_metadata("some string", "tensor")
        self.assertEqual(result["type"], "unknown")
        self.assertIn("some string", result["value"])


class TestGetMediaContentType(unittest.TestCase):
    """Tests for get_media_content_type."""

    def test_image_content_type(self):
        """12. image → image/png."""
        self.assertEqual(get_media_content_type("image"), "image/png")

    def test_unknown_content_type(self):
        """Unknown field_type → application/octet-stream."""
        self.assertEqual(
            get_media_content_type("something_else"),
            "application/octet-stream",
        )


if __name__ == "__main__":
    unittest.main()
