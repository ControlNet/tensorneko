from unittest import TestCase

import torch
from torch import rand

from tensorneko.preprocess import resize_image, resize_video, padding_video, PaddingMethod, PaddingPosition, \
    padding_audio


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

    @property
    def target_t(self):
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

    def test_padding_video_same(self):
        # test video x (T, C, H, W)
        video_size = torch.Size((self.t, self.c, self.h, self.w))
        x = rand(video_size)
        # target
        padded = padding_video(x, self.target_t, padding_method=PaddingMethod.SAME,
            padding_position=PaddingPosition.TAIL)
        self.assertTrue(padded.shape == torch.Size((self.target_t, self.c, self.h, self.w)))
        self.assertTrue(torch.equal(padded[-1], x[-1]))
        self.assertTrue(torch.equal(padded[self.t], x[-1]))

    def test_padding_video_zero(self):
        # test video x (T, C, H, W)
        video_size = torch.Size((self.t, self.c, self.h, self.w))
        x = rand(video_size)
        # target
        padded = padding_video(x, self.target_t, padding_method=PaddingMethod.ZERO,
            padding_position=PaddingPosition.TAIL)
        self.assertTrue(padded.shape == torch.Size((self.target_t, self.c, self.h, self.w)))
        self.assertTrue(torch.equal(padded[-1], torch.zeros(self.c, self.h, self.w)))
        self.assertTrue(torch.equal(padded[self.t], torch.zeros(self.c, self.h, self.w)))

    def test_padding_audio(self):
        # test audio x (T, 2)
        n_channels = 2
        audio_size = torch.Size((self.t, n_channels))
        x = rand(audio_size)
        # target
        padded = padding_audio(x, self.target_t, padding_method=PaddingMethod.SAME,
            padding_position=PaddingPosition.TAIL)
        self.assertTrue(padded.shape == torch.Size((self.target_t, n_channels)))
        self.assertTrue(torch.equal(padded[-1], x[-1]))
        self.assertTrue(torch.equal(padded[self.t], x[-1]))