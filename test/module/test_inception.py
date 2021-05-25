import unittest

import torch
from torch import rand, nn, cat

from tensorneko.module import InceptionModule


class TestInceptionModule(unittest.TestCase):

    def test_output_shape(self):
        # generate test input
        x = rand(50, 24)
        # build inception module
        model = InceptionModule().add_sub_sequence(
            nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, 100)
        ).add_sub_sequence(
            nn.Linear(24, 12), nn.Sigmoid(), nn.Linear(12, 12)
        )
        # test shape size
        self.assertEqual(model(x).shape, torch.Size((50, 100 + 12)))

    def test_forward(self):
        # generate test input
        x = rand(50, 24)
        # build inception module
        model = InceptionModule().add_sub_sequence(
            nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, 100)
        ).add_sub_sequence(
            nn.Linear(24, 12), nn.Sigmoid(), nn.Linear(12, 12)
        )
        # test shape size
        neko_res = model(x)
        pl_res = cat([model.sub_seqs[0](x), model.sub_seqs[1](x)], dim=1)
        self.assertTrue((neko_res - pl_res).sum() < 1e-8)
