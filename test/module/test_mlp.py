import unittest

import torch
from fn import F
from torch.nn import Linear, Sigmoid, ELU, LayerNorm

from tensorneko.module import MLP


class TestMLP(unittest.TestCase):

    def test_purely_linear_layers(self):
        neurons = [128, 256, 256, 1]
        batch = 4
        mlp = MLP(neurons, bias=False)
        # checking the linear structure is in MLP
        self.assertEqual(str(mlp.layers[0].linear), str(Linear(neurons[0], neurons[1], bias=False)))
        self.assertEqual(str(mlp.layers[1].linear), str(Linear(neurons[1], neurons[2], bias=False)))
        self.assertEqual(str(mlp.layers[2].linear), str(Linear(neurons[2], neurons[3], bias=False)))
        # checking forward computation graph
        x = torch.rand(batch, neurons[0])  # four 128-dim vectors
        pt_res = mlp(x)
        neko_res = (F() >> mlp.layers[0].linear >> mlp.layers[1].linear >> mlp.layers[2].linear)(x)
        for i in range(batch):
            self.assertAlmostEqual(pt_res[i], neko_res[i])

    def test_mlp_with_sigmoid(self):
        neurons = [128, 256, 256, 1]
        batch = 4
        mlp = MLP(neurons, build_activation=Sigmoid)
        # checking the linear structure is in MLP
        self.assertEqual(str(mlp.layers[0].activation), str(Sigmoid()))
        self.assertEqual(str(mlp.layers[1].activation), str(Sigmoid()))
        self.assertEqual(str(mlp.layers[2].activation), str(None))
        # checking forward computation graph
        x = torch.rand(batch, neurons[0])  # four 128-dim vectors
        pt_res = mlp(x)
        neko_res = (
            F() >>
            mlp.layers[0].linear >> mlp.layers[0].activation >>
            mlp.layers[1].linear >> mlp.layers[1].activation >>
            mlp.layers[2].linear
        )(x)
        for i in range(batch):
            self.assertAlmostEqual(pt_res[i], neko_res[i])

    def test_mlp_with_activation_after_normalization(self):
        # build MLP
        neurons = [128, 256, 256, 1]
        batch = 4
        mlp = MLP(neurons, build_activation=ELU,
            build_normalization=[
                F(LayerNorm, [batch, neurons[1]]), F(LayerNorm, [batch, neurons[2]])
            ]
        )
        # checking the linear structure is in MLP
        self.assertEqual(str(mlp.layers[0].normalization), str(LayerNorm([batch, neurons[1]])))
        self.assertEqual(str(mlp.layers[1].normalization), str(LayerNorm([batch, neurons[2]])))
        self.assertEqual(str(mlp.layers[2].normalization), str(None))
        # checking forward computation graph
        x = torch.rand(batch, neurons[0])  # four 128-dim vectors
        pt_res = mlp(x)
        neko_res = (
            F() >>
            mlp.layers[0].linear >> mlp.layers[0].normalization >> mlp.layers[0].activation >>
            mlp.layers[1].linear >> mlp.layers[1].normalization >> mlp.layers[1].activation >>
            mlp.layers[2].linear
        )(x)
        for i in range(batch):
            self.assertAlmostEqual(pt_res[i], neko_res[i])

    def test_mlp_with_activation_before_normalization(self):
        # build MLP
        neurons = [128, 256, 256, 1]
        batch = 4
        mlp = MLP(neurons, build_activation=ELU,
            build_normalization=[
                F(LayerNorm, [batch, neurons[1]]), F(LayerNorm, [batch, neurons[2]])
            ], normalization_after_activation=True
        )
        # checking forward computation graph
        x = torch.rand(batch, neurons[0])  # four 128-dim vectors
        pt_res = mlp(x)
        neko_res = (
            F() >>
            mlp.layers[0].linear >> mlp.layers[0].activation >> mlp.layers[0].normalization >>
            mlp.layers[1].linear >> mlp.layers[1].activation >> mlp.layers[1].normalization >>
            mlp.layers[2].linear
        )(x)
        for i in range(batch):
            self.assertAlmostEqual(pt_res[i], neko_res[i])
