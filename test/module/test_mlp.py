import unittest

import torch
from torch.nn import Linear, Sigmoid, ELU, LayerNorm

from tensorneko.module import MLP
from tensorneko.util import F


class TestMLP(unittest.TestCase):

    @property
    def neurons(self):
        return [128, 256, 256, 10]

    @property
    def batch(self):
        return 4

    def test_purely_linear_layers(self):
        mlp = MLP(self.neurons, bias=False)
        # checking the linear structure is in MLP
        self.assertEqual(str(mlp.layers[0].linear), str(Linear(self.neurons[0], self.neurons[1], bias=False)))
        self.assertEqual(str(mlp.layers[1].linear), str(Linear(self.neurons[1], self.neurons[2], bias=False)))
        self.assertEqual(str(mlp.layers[2].linear), str(Linear(self.neurons[2], self.neurons[3], bias=False)))
        # checking forward computation graph
        x = torch.rand(self.batch, self.neurons[0])
        neko_res = mlp(x)
        pt_res = (F() >> mlp.layers[0].linear >> mlp.layers[1].linear >> mlp.layers[2].linear)(x)
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)

    def test_mlp_with_sigmoid(self):
        mlp = MLP(self.neurons, build_activation=Sigmoid)
        # checking the linear structure is in MLP
        self.assertEqual(str(mlp.layers[0].activation), str(Sigmoid()))
        self.assertEqual(str(mlp.layers[1].activation), str(Sigmoid()))
        self.assertEqual(str(mlp.layers[2].activation), str(None))
        # checking forward computation graph
        x = torch.rand(self.batch, self.neurons[0])
        neko_res = mlp(x)
        pt_res = (
            F() >>
            mlp.layers[0].linear >> mlp.layers[0].activation >>
            mlp.layers[1].linear >> mlp.layers[1].activation >>
            mlp.layers[2].linear
        )(x)
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)

    def test_mlp_with_activation_after_normalization(self):
        # build MLP
        mlp = MLP(self.neurons, build_activation=ELU,
            build_normalization=[
                F(LayerNorm, [self.batch, self.neurons[1]]), F(LayerNorm, [self.batch, self.neurons[2]])
            ]
        )
        # checking the linear structure is in MLP
        self.assertEqual(str(mlp.layers[0].normalization), str(LayerNorm([self.batch, self.neurons[1]])))
        self.assertEqual(str(mlp.layers[1].normalization), str(LayerNorm([self.batch, self.neurons[2]])))
        self.assertEqual(str(mlp.layers[2].normalization), str(None))
        # checking forward computation graph
        x = torch.rand(self.batch, self.neurons[0])
        neko_res = mlp(x)
        pt_res = (
            F() >>
            mlp.layers[0].linear >> mlp.layers[0].normalization >> mlp.layers[0].activation >>
            mlp.layers[1].linear >> mlp.layers[1].normalization >> mlp.layers[1].activation >>
            mlp.layers[2].linear
        )(x)
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)

    def test_mlp_with_activation_before_normalization(self):
        # build MLP
        mlp = MLP(self.neurons, build_activation=ELU,
            build_normalization=[
                F(LayerNorm, [self.batch, self.neurons[1]]), F(LayerNorm, [self.batch, self.neurons[2]])
            ], normalization_after_activation=True
        )
        # checking forward computation graph
        x = torch.rand(self.batch, self.neurons[0])
        neko_res = mlp(x)
        pt_res = (
            F() >>
            mlp.layers[0].linear >> mlp.layers[0].activation >> mlp.layers[0].normalization >>
            mlp.layers[1].linear >> mlp.layers[1].activation >> mlp.layers[1].normalization >>
            mlp.layers[2].linear
        )(x)
        self.assertTrue((pt_res - neko_res).sum() < 1e-8)
