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
        neko_res2 = (
            __()
            >> (x1, x2)
            >> (lambda x, y: x + y)
            >> (lambda x: x + 1)
            >> (lambda x: x * 2)
            >> __.get
        )
        neko_res3 = __(x1, x2) >> (_ + _) >> (_ + 1) >> (_ * 2) >> __.get
        true_res = ((x1 + x2) + 1) * 2
        self.assertEqual(neko_res1, true_res)
        self.assertEqual(neko_res2, true_res)
        self.assertEqual(neko_res3, true_res)

    def test_args_merge_getters_and_string(self):
        merged = __(1, a=2) >> __(3, b=4)
        self.assertEqual(merged.get_args(), (1, 3))
        self.assertEqual(merged.get_kwargs(), {"a": 2, "b": 4})
        self.assertEqual(merged.get_value("b"), 4)
        self.assertEqual(str(merged), "(1, 3, a=2, b=4)")

    def test_args_invalid_operators(self):
        self.assertRaises(ValueError, lambda: __(1) << (_ + 1))
        self.assertRaises(TypeError, lambda: __(1)())
