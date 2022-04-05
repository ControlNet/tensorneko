import unittest

from torch import rand
from torch.nn import Linear, ReLU

from tensorneko.module import ResidualModule, ResidualBlock
from tensorneko.util import F


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
        build_sub_module = [F(Linear, 24, 48), ReLU, F(Linear, 48, 24)]
        build_tail_module = [F(Linear, 24, 12)]
        block = ResidualBlock(build_sub_module, build_tail_module)
        # init test input
        x = rand(self.batch, self.input_dim)
        # tensorneko forward
        neko_res = block(x)
        # pytorch forward
        pt_sub_module = block.sub_module
        pt_tail_module = block.tail_module
        pt_res = pt_tail_module(pt_sub_module(x) + x)
        return self.assertTrue((pt_res - neko_res).mean().abs() < 1e-8)


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
        neko_module = ResidualModule(F(ResidualBlock, (F(Linear, 24, 48), ReLU, F(Linear, 48, 24))), self.repeat)
        # get str representation of first block
        first_block_hparams = str(neko_module.blocks[0])
        # compare other blocks to it
        for i in range(1, self.repeat):
            self.assertEqual(str(neko_module.blocks[i]), first_block_hparams)
            # but the object is not the same one
            self.assertTrue(neko_module.blocks[i].sub_module[0] is not neko_module.blocks[0].sub_module[0])

    def test_repeat_modules_has_same_forward(self):
        module = ResidualModule(F(ResidualBlock, (F(Linear, 24, 48), ReLU, F(Linear, 48, 24))), self.repeat)
        # init test input
        x = rand(self.batch, self.input_dim)
        # tensorneko forward
        neko_res = module(x)
        # pytorch forward
        pt_res = x
        for block in module.blocks:
            pt_res = block(pt_res)
        # compare
        self.assertTrue((neko_res - pt_res).mean().abs() < 1e-8)