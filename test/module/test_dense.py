import unittest

from torch import randn
from torch.nn import BatchNorm2d, ReLU

from tensorneko.layer import Conv2d
from tensorneko.module.dense import DenseBlock
from tensorneko.util import F


class TestDense(unittest.TestCase):

    @property
    def c(self):
        return 4

    @property
    def h(self):
        return 64

    @property
    def w(self):
        return 64

    @property
    def repeat(self):
        return 3

    def test_dense_block(self):
        # generate test input
        x = randn((1, self.c, self.h, self.w))

        # build dense block
        def build_bn(i=0):
            return BatchNorm2d(2 ** i * self.c)

        def build_conv2d_1x1(i=0):
            return Conv2d(2 ** i * self.c, 2 ** i * self.c * 4, (1, 1), build_activation=ReLU,
                build_normalization=F(BatchNorm2d, 2 ** i * self.c * 4))

        def build_conv2d_3x3(i=0):
            return Conv2d(2 ** i * self.c * 4, 2 ** i * self.c, (3, 3), padding=(1, 1))

        dense_block = DenseBlock((
            build_bn,
            lambda i: ReLU(),
            build_conv2d_1x1,
            build_conv2d_3x3
        ), repeat=self.repeat)

        # check each layer should not be the same object
        for i in range(1, self.repeat):
            for j in range(4):
                self.assertFalse(dense_block.sub_modules[i][j] is dense_block.sub_modules[0][j])

        # the shape should be matched
        shape = tuple(dense_block(x).shape)
        expect_shape = (1, 2 ** self.repeat * self.c, self.h, self.w)
        for i in range(len(shape)):
            self.assertEqual(shape[i], expect_shape[i])
