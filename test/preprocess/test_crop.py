import unittest
import numpy as np
import torch
from tensorneko.preprocess.crop import crop_with_padding
from tensorneko_util.preprocess.crop import crop_with_padding as crop_with_padding_numpy


class TestCropWithPadding(unittest.TestCase):
    """Test suite for crop_with_padding function."""

    def test_3d_center_crop_within_bounds(self):
        """Test 3D tensor (C, H, W): center crop entirely within bounds."""
        # Create a 3D tensor with deterministic values: (3, 4, 5)
        image = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)

        # Crop the middle 2x3 region from (1, 1) to (3, 4) in H,W
        result = crop_with_padding(image, x1=1, x2=4, y1=1, y2=3, pad_value=0.0)

        # Expected shape: (3, 2, 3)
        self.assertEqual(result.shape, (3, 2, 3))

        # Verify values from first channel
        expected_ch0 = torch.tensor([[6.0, 7.0, 8.0], [11.0, 12.0, 13.0]])
        torch.testing.assert_close(result[0], expected_ch0)

    def test_3d_crop_beyond_top_left_with_padding(self):
        """Test 3D tensor: crop partially beyond top-left corner (x1 < 0, y1 < 0)."""
        # Create a 3D tensor: (2, 3, 3)
        image = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)

        # Crop from (-1, -1) to (2, 2), pad region should be filled
        result = crop_with_padding(image, x1=-1, x2=2, y1=-1, y2=2, pad_value=0.0)

        # Expected shape: (2, 3, 3) - the crop region is 3x3
        self.assertEqual(result.shape, (2, 3, 3))

        # First row and column should be padded with 0s
        # Channel 0 layout:
        # [0  0  0]
        # [0  0  1]
        # [0  3  4]
        expected_ch0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 3.0, 4.0]])
        torch.testing.assert_close(result[0], expected_ch0)

    def test_3d_crop_beyond_bottom_right_with_padding(self):
        """Test 3D tensor: crop partially beyond bottom-right corner (x2 > W, y2 > H)."""
        # Create a 3D tensor: (1, 2, 2)
        image = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

        # Crop from (0, 0) to (3, 3), which extends beyond image bounds
        result = crop_with_padding(image, x1=0, x2=3, y1=0, y2=3, pad_value=0.0)

        # Expected shape: (1, 3, 3)
        self.assertEqual(result.shape, (1, 3, 3))

        # Padded region should be on bottom and right
        # [0  1  0]
        # [2  3  0]
        # [0  0  0]
        expected = torch.tensor([[0.0, 1.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(result[0], expected)

    def test_3d_crop_fully_out_of_bounds_all_padding(self):
        """Test 3D tensor: crop region entirely outside image bounds (all padding)."""
        # Create a 3D tensor: (1, 2, 2)
        image = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

        # Crop from (5, 5) to (8, 8), completely outside the image
        result = crop_with_padding(image, x1=5, x2=8, y1=5, y2=8, pad_value=0.0)

        # Expected shape: (1, 3, 3) with all zeros
        self.assertEqual(result.shape, (1, 3, 3))
        expected = torch.zeros(1, 3, 3)
        torch.testing.assert_close(result, expected)

    def test_4d_batch_crop(self):
        """Test 4D tensor (B, C, H, W): batch dimension preserved."""
        # Create a batch of 2 images: (2, 1, 3, 3)
        image = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)

        # Crop from (0, 0) to (2, 2)
        result = crop_with_padding(
            image, x1=0, x2=2, y1=0, y2=2, pad_value=0.0, batch=True
        )

        # Expected shape: (2, 1, 2, 2) - batch and channel dimensions preserved
        self.assertEqual(result.shape, (2, 1, 2, 2))

        # Verify first batch item
        expected_0 = torch.tensor([[0.0, 1.0], [3.0, 4.0]]).unsqueeze(
            0
        )  # Add channel dimension
        torch.testing.assert_close(result[0], expected_0)

        # Verify second batch item
        expected_1 = torch.tensor([[9.0, 10.0], [12.0, 13.0]]).unsqueeze(0)
        torch.testing.assert_close(result[1], expected_1)

    def test_2d_tensor_without_channel(self):
        """Test 2D tensor (H, W): without channel dimension."""
        # Create a 2D tensor: (3, 4)
        image = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        # Crop from (1, 1) to (3, 3)
        result = crop_with_padding(image, x1=1, x2=3, y1=1, y2=3, pad_value=0.0)

        # Expected shape: (2, 2) - no channel dimension
        self.assertEqual(result.shape, (2, 2))

        # Verify values
        expected = torch.tensor([[5.0, 6.0], [9.0, 10.0]])
        torch.testing.assert_close(result, expected)

    def test_custom_pad_value(self):
        """Test custom padding value (not 0)."""
        # Create a 3D tensor: (1, 2, 2)
        image = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)

        # Crop with pad_value = -1.0
        result = crop_with_padding(image, x1=-1, x2=2, y1=-1, y2=2, pad_value=-1.0)

        # Expected shape: (1, 3, 3)
        self.assertEqual(result.shape, (1, 3, 3))

        # Check padded regions are -1.0
        expected = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 2.0, 3.0]]
        )
        torch.testing.assert_close(result[0], expected)

    def test_invalid_crop_bounds_raises_value_error(self):
        image = torch.ones(1, 2, 2)

        with self.assertRaises(ValueError):
            crop_with_padding(image, x1=2, x2=2, y1=0, y2=1)

        with self.assertRaises(ValueError):
            crop_with_padding(image, x1=0, x2=1, y1=1, y2=1)

    def test_value_error_invalid_shape(self):
        """Test ValueError for invalid tensor shapes."""
        # Invalid shape: 5D tensor
        image = torch.ones(1, 1, 1, 2, 2)

        with self.assertRaises(ValueError):
            crop_with_padding(image, x1=0, x2=1, y1=0, y2=1)

    def test_dtype_preservation(self):
        """Test that dtype is preserved after cropping."""
        # Create a 3D tensor with float64
        image = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)

        result = crop_with_padding(image, x1=0, x2=2, y1=0, y2=2, pad_value=0.0)

        # Check dtype is preserved
        self.assertEqual(result.dtype, torch.float64)


class TestCropWithPaddingNumpy(unittest.TestCase):
    def test_numpy_crop_with_partial_out_of_bounds_padding(self):
        image = np.arange(12, dtype=np.float32).reshape(3, 4)

        result = crop_with_padding_numpy(image, x1=-1, x2=3, y1=1, y2=4, pad_value=-1.0)

        expected = np.array(
            [
                [-1.0, 4.0, 5.0, 6.0],
                [-1.0, 8.0, 9.0, 10.0],
                [-1.0, -1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_allclose(result, expected)

    def test_numpy_batch_color_image_shape_and_values(self):
        image = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)

        result = crop_with_padding_numpy(image, x1=1, x2=4, y1=0, y2=2, batch=True)

        self.assertEqual(result.shape, (2, 2, 3, 2))
        np.testing.assert_allclose(result[0], image[0, 0:2, 1:4, :])
        np.testing.assert_allclose(result[1], image[1, 0:2, 1:4, :])

    def test_numpy_invalid_shape_raises_value_error(self):
        image = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)

        with self.assertRaises(ValueError):
            crop_with_padding_numpy(image, x1=0, x2=1, y1=0, y2=1)

    def test_numpy_scalar_coordinate_dispatch(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3)

        result = crop_with_padding_numpy(
            image,
            x1=np.array(1),
            x2=np.array(3),
            y1=np.array(0),
            y2=np.array(2),
        )

        expected = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_numpy_invalid_bounds_raise_value_error(self):
        image = np.ones((3, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            crop_with_padding_numpy(image, x1=1, x2=1, y1=0, y2=2)


if __name__ == "__main__":
    unittest.main()
