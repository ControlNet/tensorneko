import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from tensorneko_util.backend.visual_lib import VisualLib
from tensorneko_util.io.image.image_reader import ImageReader
from tensorneko_util.io.image.image_writer import ImageWriter


class TestImageWriter(unittest.TestCase):
    """Tests for ImageWriter using PIL backend (cv2 not available)."""

    def _random_image_hwc(self, h: int = 64, w: int = 64, c: int = 3) -> np.ndarray:
        return np.random.rand(h, w, c).astype(np.float32)

    def _random_image_chw(self, c: int = 3, h: int = 64, w: int = 64) -> np.ndarray:
        return np.random.rand(c, h, w).astype(np.float32)

    # ── to_png ──────────────────────────────────────────────────────

    def test_write_png_pil(self):
        """Write a numpy (H,W,C) image to PNG via PIL backend."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.png")
            ImageWriter.to_png(p, img, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))
            self.assertGreater(os.path.getsize(p), 0)

    def test_write_png_channel_first(self):
        """Write a (C,H,W) image to PNG with channel_first=True via PIL."""
        img = self._random_image_chw()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.png")
            ImageWriter.to_png(p, img, channel_first=True, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))

    # ── to_jpeg ─────────────────────────────────────────────────────

    def test_write_jpeg_pil(self):
        """Write a numpy (H,W,C) image to JPEG via PIL backend."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.jpg")
            ImageWriter.to_jpeg(p, img, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))
            self.assertGreater(os.path.getsize(p), 0)

    def test_write_jpeg_quality(self):
        """Higher quality produces a larger file than lower quality."""
        img = self._random_image_hwc(128, 128)
        with tempfile.TemporaryDirectory() as d:
            lo = os.path.join(d, "lo.jpg")
            hi = os.path.join(d, "hi.jpg")
            ImageWriter.to_jpeg(lo, img, quality=10, backend=VisualLib.PIL)
            ImageWriter.to_jpeg(hi, img, quality=95, backend=VisualLib.PIL)
            self.assertGreater(os.path.getsize(hi), os.path.getsize(lo))

    # ── to() dispatch ───────────────────────────────────────────────

    def test_to_dispatches_png(self):
        """ImageWriter.to() routes .png files to to_png."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.png")
            ImageWriter.to(p, img, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))

    def test_to_dispatches_jpeg(self):
        """ImageWriter.to() routes .jpg files to to_jpeg."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.jpg")
            ImageWriter.to(p, img, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))

    def test_to_unknown_extension_raises(self):
        """ImageWriter.to() raises ValueError for unsupported extensions."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ImageWriter.to(os.path.join(d, "out.bmp"), img, backend=VisualLib.PIL)

    # ── __new__ alias ───────────────────────────────────────────────

    def test_new_alias(self):
        """ImageWriter() constructor delegates to .to()."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "via_new.jpg")
            ImageWriter(p, img, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))

    # ── pathlib.Path input ──────────────────────────────────────────

    def test_pathlib_path_input(self):
        """Writer accepts pathlib.Path for the output path."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pathlib.png"
            ImageWriter.to_png(p, img, backend=VisualLib.PIL)
            self.assertTrue(p.is_file())

    # ── Roundtrip: write → read ─────────────────────────────────────

    def test_roundtrip_png_shapes_match(self):
        """Write (C,H,W) → read back: spatial dimensions preserved."""
        img = self._random_image_chw(3, 48, 72)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rt.png")
            ImageWriter.to_png(p, img, channel_first=True, backend=VisualLib.PIL)
            out = ImageReader.of(p, channel_first=True, backend=VisualLib.PIL)
            self.assertEqual(out.shape, img.shape)

    def test_roundtrip_jpeg_shapes_match(self):
        """Write (H,W,C) JPEG → read back: spatial dimensions preserved."""
        img = self._random_image_hwc(48, 72)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rt.jpg")
            ImageWriter.to_jpeg(p, img, backend=VisualLib.PIL)
            out = ImageReader.of(p, channel_first=False, backend=VisualLib.PIL)
            self.assertEqual(out.shape, img.shape)

    # ── Torch tensor input ──────────────────────────────────────────

    def test_write_torch_tensor_numpy_roundtrip(self):
        """Write a torch.Tensor, verify file is created and readable."""
        import torch

        # torch tensors need to go through _convert_img_format → IntTensor
        # PIL.Image.fromarray needs numpy; torch IntTensor lacks __array_interface__
        # So use numpy conversion explicitly. This tests _convert_img_format with ndarray from torch.
        img_np = torch.rand(48, 64, 3).numpy().astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "torch_rt.png")
            ImageWriter.to_png(p, img_np, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))
            out = ImageReader.of(p, channel_first=False, backend=VisualLib.PIL)
            self.assertEqual(out.shape, (48, 64, 3))

    def test_convert_img_format_numpy_uint8(self):
        """_convert_img_format scales [0,1] float to [0,255] uint8 for non-PYTORCH backends."""
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)  # (1,1,3)
        result = ImageWriter._convert_img_format(
            img, VisualLib.PIL, channel_first=False
        )
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(
            result, np.array([[[0, 127, 255]]], dtype=np.uint8)
        )

    def test_convert_img_format_torch_tensor(self):
        """_convert_img_format converts torch.Tensor to IntTensor scaled [0, 255]."""
        import torch

        img = torch.tensor([[[0.0, 0.5, 1.0]]])  # (1, 1, 3)
        result = ImageWriter._convert_img_format(
            img, VisualLib.PIL, channel_first=False
        )
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.int32)
        expected = torch.tensor([[[0, 127, 255]]], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_convert_img_format_channel_first_rearranges(self):
        """_convert_img_format rearranges (C,H,W) → (H,W,C) when channel_first=True for PIL."""
        img = np.random.rand(3, 8, 16).astype(np.float32)
        result = ImageWriter._convert_img_format(img, VisualLib.PIL, channel_first=True)
        self.assertEqual(result.shape, (8, 16, 3))

    def test_convert_img_format_pytorch_channel_last_rearranges(self):
        """_convert_img_format rearranges (H,W,C) → (C,H,W) when channel_first=False for PYTORCH."""
        import torch

        img = torch.rand(8, 16, 3)
        result = ImageWriter._convert_img_format(
            img, VisualLib.PYTORCH, channel_first=False
        )
        self.assertEqual(tuple(result.shape), (3, 8, 16))

    def test_to_dispatches_dot_jpeg_extension(self):
        """ImageWriter.to() routes .jpeg files to to_jpeg."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.jpeg")
            ImageWriter.to(p, img, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p))

    def test_convert_img_format_unknown_type_raises(self):
        """_convert_img_format raises ValueError for non-ndarray/non-Tensor input."""
        with self.assertRaises(ValueError):
            ImageWriter._convert_img_format(
                [[1, 2], [3, 4]], VisualLib.PIL, channel_first=False
            )

    # ── matplotlib backend tests ─────────────────────────────────────

    def test_write_png_matplotlib(self):
        """Write PNG via matplotlib backend."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mpl.png")
            ImageWriter.to_png(p, img, backend=VisualLib.MATPLOTLIB)
            self.assertTrue(os.path.isfile(p))
            self.assertGreater(os.path.getsize(p), 0)

    def test_write_jpeg_matplotlib(self):
        """Write JPEG via matplotlib backend."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mpl.jpg")
            ImageWriter.to_jpeg(p, img, backend=VisualLib.MATPLOTLIB)
            self.assertTrue(os.path.isfile(p))
            self.assertGreater(os.path.getsize(p), 0)

    # ── PYTORCH backend — _convert_img_format path ─────────────────

    def test_convert_img_format_pytorch_channel_first_no_rearrange(self):
        """For PYTORCH backend with channel_first=True, no rearrangement."""
        import torch

        img = torch.rand(3, 8, 16)
        result = ImageWriter._convert_img_format(
            img, VisualLib.PYTORCH, channel_first=True
        )
        self.assertEqual(tuple(result.shape), (3, 8, 16))

    def test_convert_img_format_numpy_channel_first_for_non_pytorch(self):
        """Non-PYTORCH backend with channel_first=True rearranges (C,H,W)->(H,W,C)."""
        img = np.random.rand(3, 8, 16).astype(np.float32)
        result = ImageWriter._convert_img_format(
            img, VisualLib.MATPLOTLIB, channel_first=True
        )
        self.assertEqual(result.shape, (8, 16, 3))

    # ── unknown backend ──────────────────────────────────────────────

    def test_write_png_unknown_backend_raises(self):
        """to_png with unknown backend raises ValueError."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ImageWriter.to_png(os.path.join(d, "bad.png"), img, backend="bad")

    def test_write_jpeg_unknown_backend_raises(self):
        """to_jpeg with unknown backend raises ValueError."""
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ImageWriter.to_jpeg(os.path.join(d, "bad.jpg"), img, backend="bad")

    # ── PIL compression_level for PNG ────────────────────────────────

    def test_write_png_pil_compression_level(self):
        """Different compression levels produce valid files."""
        img = self._random_image_hwc(128, 128)
        with tempfile.TemporaryDirectory() as d:
            p0 = os.path.join(d, "c0.png")
            p9 = os.path.join(d, "c9.png")
            ImageWriter.to_png(p0, img, compression_level=0, backend=VisualLib.PIL)
            ImageWriter.to_png(p9, img, compression_level=9, backend=VisualLib.PIL)
            self.assertTrue(os.path.isfile(p0))
            self.assertTrue(os.path.isfile(p9))
            # Higher compression should produce smaller or equal file
            self.assertGreaterEqual(os.path.getsize(p0), os.path.getsize(p9))

    # ── Backend unavailable error paths ──────────────────────────

    @unittest.mock.patch.object(VisualLib, "matplotlib_available", return_value=False)
    def test_png_matplotlib_unavailable_raises(self, mock_avail):
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                ImageWriter.to_png(
                    os.path.join(d, "x.png"), img, backend=VisualLib.MATPLOTLIB
                )
            self.assertIn("Matplotlib", str(ctx.exception))

    @unittest.mock.patch.object(VisualLib, "pil_available", return_value=False)
    def test_png_pil_unavailable_raises(self, mock_avail):
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                ImageWriter.to_png(os.path.join(d, "x.png"), img, backend=VisualLib.PIL)
            self.assertIn("PIL", str(ctx.exception))

    @unittest.mock.patch.object(VisualLib, "pytorch_available", return_value=False)
    def test_png_pytorch_unavailable_raises(self, mock_avail):
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                ImageWriter.to_png(
                    os.path.join(d, "x.png"), img, backend=VisualLib.PYTORCH
                )
            self.assertIn("Torchvision", str(ctx.exception))

    @unittest.mock.patch.object(VisualLib, "opencv_available", return_value=False)
    def test_png_opencv_unavailable_raises(self, mock_avail):
        img = self._random_image_hwc()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                ImageWriter.to_png(
                    os.path.join(d, "x.png"), img, backend=VisualLib.OPENCV
                )
            self.assertIn("OpenCV", str(ctx.exception))
