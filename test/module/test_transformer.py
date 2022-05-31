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

class TestTransformerEncoder(unittest.TestCase):

    def test_shape(self):
        shape = torch.Size((8, 16, 128))
        x = rand(shape)
        encoder = TransformerEncoder((16, 128), 4, repeat=5)
        self.assertTrue(encoder(x).shape == shape)