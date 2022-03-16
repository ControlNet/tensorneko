import random
import unittest

import numpy as np
import torch

from tensorneko.util import Seed


class UtilReproducibilityTest(unittest.TestCase):

    def test_seed(self):
        Seed.set(42)
        np_result = np.random.rand(10)
        torch_result = torch.rand(10)
        py_result = random.random()
        Seed.set(42)
        self.assertTrue(np.equal(np_result, np.random.rand(10)).all())
        self.assertTrue(torch.equal(torch_result, torch.rand(10)))
        self.assertEqual(random.random(), py_result)


if __name__ == '__main__':
    unittest.main()
