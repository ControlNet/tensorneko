import unittest
from unittest.mock import patch

import numpy as np
from torch import rand

from tensorneko_util.backend import VisualLib
from tensorneko_util.preprocess.image import rgb2gray, rgb2gray_batch


class TestImagePreprocess(unittest.TestCase):
    def test_rgb2gray_pil_with_float_input(self):
        img = rand(12, 16, 3).numpy()

        out = rgb2gray(img, backend=VisualLib.PIL)

        self.assertEqual(out.shape, (12, 16))
        self.assertTrue(np.issubdtype(out.dtype, np.floating))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_rgb2gray_channel_first_matches_channel_last(self):
        channel_last = rand(10, 14, 3).numpy()
        channel_first = np.transpose(channel_last, (2, 0, 1))

        expected = rgb2gray(channel_last, channel_first=False, backend=VisualLib.PIL)
        out = rgb2gray(channel_first, channel_first=True, backend=VisualLib.PIL)

        np.testing.assert_allclose(out, expected)

    def test_rgb2gray_batch_channel_first(self):
        batch = rand(4, 3, 8, 9).numpy()

        out = rgb2gray_batch(batch, channel_first=True, backend=VisualLib.PIL)

        self.assertEqual(out.shape, (4, 8, 9))
        expected = np.asarray(
            [rgb2gray(img.transpose(1, 2, 0), backend=VisualLib.PIL) for img in batch]
        )
        np.testing.assert_allclose(out, expected)

    def test_rgb2gray_opencv_backend_unavailable(self):
        img = rand(6, 7, 3).numpy()

        with patch.object(VisualLib, "opencv_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "OpenCV is not installed"):
                rgb2gray(img, backend=VisualLib.OPENCV)

    def test_rgb2gray_pil_backend_unavailable(self):
        img = rand(6, 7, 3).numpy()

        with patch.object(VisualLib, "pil_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Pillow is not installed"):
                rgb2gray(img, backend=VisualLib.PIL)

    def test_rgb2gray_skimage_backend_unavailable(self):
        img = rand(6, 7, 3).numpy()

        with patch.object(VisualLib, "skimage_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "scikit-image is not installed"):
                rgb2gray(img, backend=VisualLib.SKIMAGE)

    def test_rgb2gray_unsupported_backend_raises(self):
        img = rand(5, 6, 3).numpy()

        with self.assertRaisesRegex(ValueError, "is not supported"):
            rgb2gray(img, backend=None)

    def test_rgb2gray_batch_without_channel_first(self):
        batch = rand(3, 7, 11, 3).numpy()

        out = rgb2gray_batch(batch, channel_first=False, backend=VisualLib.PIL)

        self.assertEqual(out.shape, (3, 7, 11))
        self.assertTrue(np.issubdtype(out.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
