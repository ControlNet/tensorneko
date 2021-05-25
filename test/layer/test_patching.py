import unittest

from torch import rand

from tensorneko.layer import PatchEmbedding2d


class TestPatching(unittest.TestCase):
    # TODO
    pass


class TestPatchEmbedding2d(unittest.TestCase):

    def test_simple_patching(self):
        # test input for 64x64 RGB image batches
        b, c, h, w = (8, 3, 64, 64)
        x = rand(b, c, h, w)
        # patch size
        p = 16
        # embedding output
        e = 512
        # build layer
        patch_layer = PatchEmbedding2d((c, h, w), p, e)
        # patch grid size
        seq_length = (h // p) * (w // p)
        self.assertTrue(patch_layer(x).shape == (b, seq_length, e))

    def test_overlap_patching(self):
        # test input for 64x64 RGB image batches
        b, c, h, w = (8, 3, 64, 64)
        x = rand(b, c, h, w)
        # patch size
        p = 16
        # embedding output
        e = 512
        # strides
        s = 8
        # build layer
        patch_layer = PatchEmbedding2d((c, h, w), p, e, strides=(s, s))
        # patch grid size
        seq_length = ((h - p) // s + 1) * ((w - p) // s + 1)
        self.assertTrue(patch_layer(x).shape == (b, seq_length, e))
