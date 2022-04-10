import unittest

import torch
from tensorneko.layer import Reshape


class TestReshape(unittest.TestCase):
    """The class for test Reshape layer"""

    def test_reshape(self):
        """The function for test Reshape layer"""
        x = torch.randn(1, 6)
        layer = Reshape((3, 2))
        neko_result = layer.forward(x)
        pytorch_result = x.reshape(3, 2)

        self.assertTrue((neko_result == pytorch_result).all())
