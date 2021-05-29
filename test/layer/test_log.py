import unittest

import torch

from tensorneko.layer import Log
from tensorneko.util import is_bad_num


class TestLog(unittest.TestCase):

    def test_log_computation(self):
        # generate test input
        x = torch.rand((1, 32)) + 1e-8
        # calculate log result for tensorneko and pytorch solution
        log = Log()
        neko_res = log(x)
        pt_res = torch.log(x)
        # assert these 2 are equal
        self.assertTrue((neko_res == pt_res).all())

    def test_log_zero_avoiding_nan(self):
        # generate test input
        x = torch.zeros((1, 32))
        log = Log(eps=1e-7)
        # assert not be NaN
        self.assertTrue(is_bad_num(torch.log(x)).any())
        self.assertFalse(is_bad_num(log(x)).any())

