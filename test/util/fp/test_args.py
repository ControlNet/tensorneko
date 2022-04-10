import unittest

from tensorneko_util.util import __, _


class UtilArgsTest(unittest.TestCase):

    def test_args(self):
        # simple forward
        x = 100
        neko_res = __(x) >> (_ + 1) >> (_ * 2) >> __.get
        true_res = (x + 1) * 2
        self.assertEqual(neko_res, true_res)
        # compose args to forward
        x1 = 100
        x2 = 20
        neko_res1 = __(x1) >> (x2,) >> (_ + _) >> (_ + 1) >> (_ * 2) >> __.get
        neko_res2 = __() >> (x1, x2) >> (lambda x, y: x + y) >> (lambda x: x + 1) >> (lambda x: x * 2) >> __.get
        neko_res3 = __(x1, x2) >> (_ + _) >> (_ + 1) >> (_ * 2) >> __.get
        true_res = ((x1 + x2) + 1) * 2
        self.assertEqual(neko_res1, true_res)
        self.assertEqual(neko_res2, true_res)
        self.assertEqual(neko_res3, true_res)
