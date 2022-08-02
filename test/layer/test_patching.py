import unittest

from torch import rand

from tensorneko.layer import PatchEmbedding2d, PatchEmbedding3d


class TestPatching(unittest.TestCase):
    # TODO
    pass


class TestPatchEmbedding2d(unittest.TestCase):

    def test_simple_patching2d(self):
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

    def test_simple_patching2d_with_tuple_patch_size(self):
        # test input for 64x64 RGB image batches
        b, c, h, w = (8, 3, 64, 64)
        x = rand(b, c, h, w)
        # patch size
        p = (16, 16)
        # embedding output
        e = 512
        # build layer
        patch_layer = PatchEmbedding2d((c, h, w), p, e)
        # patch grid size
        seq_length = (h // p[0]) * (w // p[1])
        self.assertTrue(patch_layer(x).shape == (b, seq_length, e))

    def test_overlap_patching2d(self):
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

    def test_simple_patching3d(self):
        # test input for 64x64 RGB image batches
        b, c, t, h, w = (8, 3, 16, 64, 64)
        x = rand(b, c, t, h, w)
        # patch size
        p = 16
        # embedding output
        e = 512
        # build layer
        patch_layer = PatchEmbedding3d((c, t, h, w), p, e)
        # patch grid size
        seq_length = (t // p) * (h // p) * (w // p)
        self.assertTrue(patch_layer(x).shape == (b, seq_length, e))

    def test_simple_patching3d_with_tuple_patch_size(self):
        # test input for 64x64 RGB image batches
        b, c, t, h, w = (8, 3, 16, 64, 64)
        x = rand(b, c, t, h, w)
        # patch size
        p = (2, 16, 16)
        # embedding output
        e = 768
        # build layer
        patch_layer = PatchEmbedding3d((c, t, h, w), p, e)
        # patch grid size
        seq_length = (t // p[0]) * (h // p[1]) * (w // p[2])
        self.assertTrue(patch_layer(x).shape == (b, seq_length, e))

    def test_overlap_patching3d(self):
        # test input for 64x64 RGB image batches
        b, c, t, h, w = (8, 3, 16, 64, 64)
        x = rand(b, c, t, h, w)
        # patch size
        p = 16
        # embedding output
        e = 512
        # strides
        s = 8
        # build layer
        patch_layer = PatchEmbedding3d((c, t, h, w), p, e, strides=(s, s, s))
        # patch grid size
        seq_length = ((t - p) // s + 1) * ((h - p) // s + 1) * ((w - p) // s + 1)
        self.assertTrue(patch_layer(x).shape == (b, seq_length, e))
