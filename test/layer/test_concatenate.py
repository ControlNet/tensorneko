import unittest

import torch
from torch import rand

from tensorneko.layer import Concatenate


class TestConcatenate(unittest.TestCase):

    @property
    def h(self) -> int:
        """height"""
        return 20

    @property
    def w(self) -> int:
        """width"""
        return 16

    @property
    def b(self) -> int:
        """batch"""
        return 4

    @property
    def c(self) -> int:
        """channel"""
        return 32

    def test_concatenate_on_channels(self):
        # generate test input
        x1 = rand(self.b, self.c, self.h, self.w)
        x2 = rand(self.b, self.c, self.h, self.w)
        x3 = rand(self.b, self.c, self.h, self.w)
        # compare result
        neko_res = Concatenate(dim=1)([x1, x2, x3])
        pl_res = torch.cat([x1, x2, x3], dim=1)
        self.assertEqual(neko_res.shape, pl_res.shape)

    def test_concatenate_on_batches(self):
        # generate test input
        x1 = rand(self.b, self.c, self.h, self.w)
        x2 = rand(self.b, self.c, self.h, self.w)
        x3 = rand(self.b, self.c, self.h, self.w)
        # compare result
        neko_res = Concatenate(dim=0)([x1, x2, x3])
        pl_res = torch.cat([x1, x2, x3], dim=0)
        self.assertEqual(neko_res.shape, pl_res.shape)
