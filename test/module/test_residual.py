import unittest

from fn import F
from torch import rand
from torch.nn import Sequential, Linear, ReLU

from tensorneko.module import ResidualModule, ResidualBlock


class TestResidualBlock(unittest.TestCase):

    @property
    def input_dim(self):
        """input vector dimension"""
        return 24

    @property
    def batch(self):
        """number of batches"""
        return 4

    def test_forward(self):
        sub_module = Sequential(Linear(24, 48), ReLU(), Linear(48, 24))
        tail_module = Linear(24, 12)
        block = ResidualBlock(sub_module, tail_module)
        # init test input
        x = rand(self.batch, self.input_dim)
        # tensorneko forward
        neko_res = block(x)
        # pytorch forward
        pt_res = tail_module(sub_module(x) + x)
        return self.assertTrue((pt_res - neko_res).sum() < 1e-8)


class TestResidualModule(unittest.TestCase):

    @property
    def repeat(self):
        """number of repeat"""
        return 8

    @property
    def input_dim(self):
        """input vector dimension"""
        return 24

    @property
    def batch(self):
        """number of batches"""
        return 4

    def test_repeat_modules_has_same_hparams(self):
        neko_module = ResidualModule(F(Sequential, Linear(24, 48), ReLU(), Linear(48, 24)), self.repeat)
        # get str representation of first block
        first_block_hparams = str(neko_module.blocks[0])
        # compare other blocks to it
        for i in range(1, self.repeat):
            self.assertEqual(str(neko_module.blocks[i]), first_block_hparams)

    def test_repeat_modules_has_same_forward(self):
        module = ResidualModule(F(Sequential, Linear(24, 48), ReLU(), Linear(48, 24)), self.repeat)
        # init test input
        x = rand(self.batch, self.input_dim)
        # tensorneko forward
        neko_res = module(x)
        # pytorch forward
        pt_res = x
        for block in module.blocks:
            pt_res = block(pt_res)
        # compare
        self.assertTrue((neko_res - pt_res).sum() < 1e-8)