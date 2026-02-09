import unittest

import numpy as np

from tensorneko_util.backend.visual_lib import VisualLib
from tensorneko_util.preprocess.image import rgb2gray, rgb2gray_batch


class TestRgb2Gray(unittest.TestCase):
    """Test rgb2gray with PIL backend (only available backend)."""

    def test_rgb2gray_hwc(self):
        """Convert (H, W, 3) float image to grayscale."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        gray = rgb2gray(img, channel_first=False, backend=VisualLib.PIL)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (32, 32))

    def test_rgb2gray_chw(self):
        """Convert (3, H, W) float image to grayscale with channel_first=True."""
        img = np.random.rand(3, 32, 32).astype(np.float32)
        gray = rgb2gray(img, channel_first=True, backend=VisualLib.PIL)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (32, 32))

    def test_rgb2gray_uint8_range(self):
        """Convert image with values in [0, 255] uint8 range."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        gray = rgb2gray(img, channel_first=False, backend=VisualLib.PIL)
        self.assertEqual(gray.ndim, 2)
        # values should be in [0, 1] range after division
        self.assertGreaterEqual(gray.min(), 0.0)
        self.assertLessEqual(gray.max(), 1.0)

    def test_rgb2gray_opencv_not_available(self):
        """OPENCV backend should raise ValueError."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            rgb2gray(img, backend=VisualLib.OPENCV)

    def test_rgb2gray_unknown_backend_raises(self):
        """Unknown backend should raise ValueError."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            rgb2gray(img, backend="unknown")


class TestRgb2GrayBatch(unittest.TestCase):
    def test_batch_hwc(self):
        imgs = np.random.rand(4, 32, 32, 3).astype(np.float32)
        grays = rgb2gray_batch(imgs, channel_first=False, backend=VisualLib.PIL)
        self.assertEqual(grays.shape, (4, 32, 32))

    def test_batch_chw(self):
        imgs = np.random.rand(4, 3, 32, 32).astype(np.float32)
        grays = rgb2gray_batch(imgs, channel_first=True, backend=VisualLib.PIL)
        self.assertEqual(grays.shape, (4, 32, 32))


if __name__ == "__main__":
    unittest.main()
