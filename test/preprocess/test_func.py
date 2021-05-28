from unittest import TestCase

import torch
from torch import rand

from tensorneko.preprocess import resize_image, resize_video


class TestFunc(TestCase):
    @property
    def c(self):
        """Channel"""
        return 3

    @property
    def h(self):
        """Height"""
        return 80

    @property
    def w(self):
        """Width"""
        return 120

    @property
    def t(self):
        """Frames of temporal"""
        return 30

    @property
    def target_h(self):
        return 160

    @property
    def target_w(self):
        return 60

    def test_resize_image(self):
        # test image x (C, H, W)
        image_size = torch.Size((self.c, self.h, self.w))
        x = rand(image_size)
        resized = resize_image(x, (self.target_h, self.target_w))
        self.assertTrue(resized.shape == torch.Size((self.c, self.target_h, self.target_w)))

    def test_resize_video(self):
        # test video x (T, C, H, W)
        video_size = torch.Size((self.t, self.c, self.h, self.w))
        x = rand(video_size)
        resized = resize_video(x, (self.target_h, self.target_w))
        self.assertTrue(resized.shape == torch.Size((self.t, self.c, self.target_h, self.target_w)))

