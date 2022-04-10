import unittest

import torch
from tensorneko.util import F

from tensorneko.layer import Linear
from torch.nn import Linear as PtLinear, LeakyReLU, BatchNorm1d, Tanh


class TestLinear(unittest.TestCase):

    @property
    def batch(self):
        return 4

    @property
    def in_neurons(self):
        return 128

    @property
    def out_neurons(self):
        return 32

    def test_simple_linear_layer(self):
        # build layers
        neko_linear = Linear(self.in_neurons, self.out_neurons, True)
        # test definition
        self.assertEqual(str(neko_linear.linear), str(PtLinear(self.in_neurons, self.out_neurons, True)))
        self.assertIs(neko_linear.activation, None)
        self.assertIs(neko_linear.normalization, None)
        # test feedforward
        x = torch.rand(self.batch, self.in_neurons)  # four 128-dim vectors
        neko_res, pt_res = map(lambda layer: layer(x), [neko_linear, neko_linear.linear])
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)

    def test_linear_with_activation(self):
        # build layers
        neko_linear = Linear(self.in_neurons, self.out_neurons, False, build_activation=LeakyReLU)
        # test definition
        self.assertEqual(str(neko_linear.linear), str(PtLinear(self.in_neurons, self.out_neurons, False)))
        self.assertEqual(str(neko_linear.activation), str(LeakyReLU()))
        self.assertIs(neko_linear.normalization, None)
        # test feedforward
        x = torch.rand(self.batch, self.in_neurons)  # four 128-dim vectors
        neko_res, pt_res = map(lambda layer: layer(x)
            , [neko_linear, F() >> neko_linear.linear >> neko_linear.activation])
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)

    def test_linear_with_activation_after_normalization(self):
        # build layers
        neko_linear = Linear(self.in_neurons, self.out_neurons, True, build_activation=Tanh,
            build_normalization=F(BatchNorm1d, self.out_neurons),
            normalization_after_activation=False
        )
        # test definition
        self.assertEqual(str(neko_linear.linear), str(PtLinear(self.in_neurons, self.out_neurons, True)))
        self.assertEqual(str(neko_linear.activation), str(Tanh()))
        self.assertEqual(str(neko_linear.normalization), str(BatchNorm1d(self.out_neurons)))
        # test feedforward
        x = torch.rand(self.batch, self.in_neurons)  # four 128-dim vectors
        neko_res, pt_res = map(lambda layer: layer(x), [neko_linear,
            F() >> neko_linear.linear >> neko_linear.normalization >> neko_linear.activation]
        )
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)

    def test_linear_with_activation_before_normalization(self):
        # build layers
        neko_linear = Linear(self.in_neurons, self.out_neurons, True, build_activation=Tanh,
            build_normalization=F(BatchNorm1d, self.out_neurons),
            normalization_after_activation=True
        )
        # test definition
        self.assertEqual(str(neko_linear.linear), str(PtLinear(self.in_neurons, self.out_neurons, True)))
        self.assertEqual(str(neko_linear.activation), str(Tanh()))
        self.assertEqual(str(neko_linear.normalization), str(BatchNorm1d(self.out_neurons)))
        # test feedforward
        x = torch.rand(self.batch, self.in_neurons)  # four 128-dim vectors
        neko_res, pt_res = map(lambda layer: layer(x), [neko_linear,
            F() >> neko_linear.linear >> neko_linear.activation >> neko_linear.normalization]
        )
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)
