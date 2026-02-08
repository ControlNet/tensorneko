import unittest

import torch

from tensorneko.preprocess.resize import resize_video, resize_image
from tensorneko.preprocess.enum import ResizeMethod


class TestResizeVideo(unittest.TestCase):
    def test_resize_video_fast_bicubic(self):
        v = torch.rand(4, 3, 16, 16)
        result = resize_video(v, (8, 8), fast=True)
        self.assertEqual(result.shape, torch.Size([4, 3, 8, 8]))

    def test_resize_video_fast_bilinear(self):
        v = torch.rand(4, 3, 16, 16)
        result = resize_video(v, (8, 8), resize_method=ResizeMethod.BILINEAR, fast=True)
        self.assertEqual(result.shape, torch.Size([4, 3, 8, 8]))

    def test_resize_video_fast_nearest(self):
        v = torch.rand(4, 3, 16, 16)
        with self.assertRaises(ValueError):
            resize_video(v, (8, 8), resize_method=ResizeMethod.NEAREST, fast=True)

    def test_resize_video_fast_string_method(self):
        v = torch.rand(4, 3, 16, 16)
        result = resize_video(v, (8, 8), resize_method="bilinear", fast=True)
        self.assertEqual(result.shape, torch.Size([4, 3, 8, 8]))

    def test_resize_video_fast_invalid_type(self):
        v = torch.rand(4, 3, 16, 16)
        with self.assertRaises(TypeError):
            resize_video(v, (8, 8), resize_method=123, fast=True)


class TestMapResizeMethod(unittest.TestCase):
    def test_resize_image_bicubic_string(self):
        img = torch.rand(3, 16, 16)
        result = resize_image(img, (8, 8), resize_method="bicubic")
        self.assertEqual(result.shape, torch.Size([3, 8, 8]))

    def test_resize_image_bilinear_string(self):
        img = torch.rand(3, 16, 16)
        result = resize_image(img, (8, 8), resize_method="bilinear")
        self.assertEqual(result.shape, torch.Size([3, 8, 8]))

    def test_resize_image_nearest_string(self):
        img = torch.rand(3, 16, 16)
        result = resize_image(img, (8, 8), resize_method="nearest")
        self.assertEqual(result.shape, torch.Size([3, 8, 8]))

    def test_resize_image_lanczos_string(self):
        img = torch.rand(3, 16, 16)
        result = resize_image(img, (8, 8), resize_method="lanczos")
        self.assertEqual(result.shape, torch.Size([3, 8, 8]))

    def test_resize_image_box_string(self):
        img = torch.rand(3, 16, 16)
        result = resize_image(img, (8, 8), resize_method="box")
        self.assertEqual(result.shape, torch.Size([3, 8, 8]))

    def test_resize_image_invalid_type(self):
        img = torch.rand(3, 16, 16)
        with self.assertRaises(TypeError):
            resize_image(img, (8, 8), resize_method=123)


if __name__ == "__main__":
    unittest.main()
