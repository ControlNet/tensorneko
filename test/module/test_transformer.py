import unittest

import torch
from torch import rand
from tensorneko.module import TransformerEncoderBlock
from tensorneko.layer.attention import SeqAttention
from tensorneko.module.transformer import TransformerEncoder


class TestAttentionModule(unittest.TestCase):
    def test_shape(self):
        shape = torch.Size((8, 16, 128))
        x = rand(shape)
        attention = SeqAttention(128, 4)
        self.assertTrue(attention(x).shape == shape)


class TestTransformerEncoderBlock(unittest.TestCase):
    def test_shape(self):
        shape = torch.Size((8, 16, 128))
        x = rand(shape)
        encoder_layer = TransformerEncoderBlock((16, 128), 4)
        self.assertTrue(encoder_layer(x).shape == shape)


class TestTransformerEncoderBlockClsToken(unittest.TestCase):
    def test_cls_token_output_shape(self):
        x = rand(torch.Size((8, 16, 128)))
        encoder_layer = TransformerEncoderBlock((16, 128), 4, add_cls_token=True)
        output = encoder_layer(x)
        self.assertTrue(output.shape == torch.Size((8, 17, 128)))

    def test_no_normalization(self):
        x = rand(torch.Size((8, 16, 128)))
        encoder_layer = TransformerEncoderBlock((16, 128), 4, build_normalization=None)
        output = encoder_layer(x)
        self.assertTrue(output.shape == torch.Size((8, 16, 128)))


class TestTransformerEncoder(unittest.TestCase):
    def test_shape(self):
        shape = torch.Size((8, 16, 128))
        x = rand(shape)
        encoder = TransformerEncoder((16, 128), 4, repeat=5)
        self.assertTrue(encoder(x).shape == shape)

    def test_pos_encoding_first(self):
        shape = torch.Size((8, 16, 128))
        x = rand(shape)
        encoder = TransformerEncoder((16, 128), 4, pos_encoding="first", repeat=3)
        self.assertTrue(encoder(x).shape == shape)

    def test_pos_encoding_none(self):
        shape = torch.Size((8, 16, 128))
        x = rand(shape)
        encoder = TransformerEncoder((16, 128), 4, pos_encoding="none", repeat=3)
        self.assertTrue(encoder(x).shape == shape)

    def test_pos_encoding_invalid(self):
        with self.assertRaises(ValueError):
            TransformerEncoder((16, 128), 4, pos_encoding="invalid", repeat=3)
