import os
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from tensorneko_util.backend.visual_lib import VisualLib
from tensorneko_util.io.image.image_reader import ImageReader

FIXTURE = os.path.join("test", "resource", "image_sample", "1.183.jpg")


class TestImageReader(unittest.TestCase):
    """Tests for ImageReader using PIL, PYTORCH, and MATPLOTLIB backends."""

    # ── PIL backend ─────────────────────────────────────────────────

    def test_read_pil_channel_first(self):
        """PIL backend returns (C, H, W) float32 ndarray in [0, 1]."""
        img = ImageReader.of(FIXTURE, channel_first=True, backend=VisualLib.PIL)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)
        c, h, w = img.shape
        self.assertEqual(c, 3)
        self.assertGreater(h, 0)
        self.assertGreater(w, 0)
        self.assertEqual(img.dtype, np.float32)
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertLessEqual(img.max(), 1.0)

    def test_read_pil_channel_last(self):
        """PIL backend with channel_first=False returns (H, W, C)."""
        img = ImageReader.of(FIXTURE, channel_first=False, backend=VisualLib.PIL)
        self.assertEqual(img.ndim, 3)
        h, w, c = img.shape
        self.assertEqual(c, 3)
        self.assertGreater(h, 0)
        self.assertGreater(w, 0)

    def test_read_pil_pathlib(self):
        """PIL backend accepts pathlib.Path input."""
        path = Path(FIXTURE)
        img = ImageReader.of(path, channel_first=True, backend=VisualLib.PIL)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[0], 3)

    # ── PYTORCH backend ─────────────────────────────────────────────

    def test_read_pytorch_channel_first(self):
        """PYTORCH backend returns (C, H, W) float torch.Tensor in [0, 1]."""
        import torch

        img = ImageReader.of(FIXTURE, channel_first=True, backend=VisualLib.PYTORCH)
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.dim(), 3)
        c, h, w = img.shape
        self.assertEqual(c, 3)
        self.assertGreaterEqual(img.min().item(), 0.0)
        self.assertLessEqual(img.max().item(), 1.0)

    def test_read_pytorch_channel_last(self):
        """PYTORCH backend with channel_first=False rearranges to (H, W, C)."""
        img = ImageReader.of(FIXTURE, channel_first=False, backend=VisualLib.PYTORCH)
        self.assertEqual(img.dim(), 3)
        h, w, c = img.shape
        self.assertEqual(c, 3)

    # ── MATPLOTLIB backend ──────────────────────────────────────────

    def test_read_matplotlib_channel_first(self):
        """MATPLOTLIB backend returns (C, H, W) ndarray."""
        img = ImageReader.of(FIXTURE, channel_first=True, backend=VisualLib.MATPLOTLIB)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[0], 3)
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertLessEqual(img.max(), 1.0)

    def test_read_matplotlib_channel_last(self):
        """MATPLOTLIB backend with channel_first=False returns (H, W, C)."""
        img = ImageReader.of(FIXTURE, channel_first=False, backend=VisualLib.MATPLOTLIB)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)

    # ── __new__ alias ───────────────────────────────────────────────

    def test_new_alias(self):
        """ImageReader() constructor is alias of .of() with channel_first=True."""
        img = ImageReader(FIXTURE, backend=VisualLib.PIL)
        self.assertEqual(img.ndim, 3)
        # __new__ defaults channel_first=True
        self.assertEqual(img.shape[0], 3)

    # ── Error paths ─────────────────────────────────────────────────

    def test_nonexistent_file_pil_raises(self):
        """Reading a non-existent file with PIL raises an error."""
        with self.assertRaises(Exception):
            ImageReader.of("nonexistent/path/img.jpg", backend=VisualLib.PIL)

    def test_unknown_backend_raises(self):
        """Passing an invalid backend value raises ValueError."""
        with self.assertRaises(ValueError):
            ImageReader.of(FIXTURE, backend="not_a_backend")

    # ── Shape consistency across backends ───────────────────────────

    def test_shapes_consistent_across_backends(self):
        """PIL, PYTORCH and MATPLOTLIB all produce the same spatial dimensions."""
        shapes = {}
        for backend in (VisualLib.PIL, VisualLib.PYTORCH, VisualLib.MATPLOTLIB):
            img = ImageReader.of(FIXTURE, channel_first=True, backend=backend)
            # Normalise to tuple of ints
            shapes[backend] = tuple(int(d) for d in img.shape)
        self.assertEqual(shapes[VisualLib.PIL], shapes[VisualLib.PYTORCH])
        self.assertEqual(shapes[VisualLib.PIL], shapes[VisualLib.MATPLOTLIB])

    # ── Backend unavailable error paths ──────────────────────────

    @unittest.mock.patch.object(VisualLib, "matplotlib_available", return_value=False)
    def test_matplotlib_unavailable_raises(self, mock_avail):
        with self.assertRaises(ValueError) as ctx:
            ImageReader.of(FIXTURE, backend=VisualLib.MATPLOTLIB)
        self.assertIn("Matplotlib", str(ctx.exception))

    @unittest.mock.patch.object(VisualLib, "pytorch_available", return_value=False)
    def test_pytorch_unavailable_raises(self, mock_avail):
        with self.assertRaises(ValueError) as ctx:
            ImageReader.of(FIXTURE, backend=VisualLib.PYTORCH)
        self.assertIn("Torchvision", str(ctx.exception))

    @unittest.mock.patch.object(VisualLib, "pil_available", return_value=False)
    def test_pil_unavailable_raises(self, mock_avail):
        with self.assertRaises(ValueError) as ctx:
            ImageReader.of(FIXTURE, backend=VisualLib.PIL)
        self.assertIn("PIL", str(ctx.exception))
