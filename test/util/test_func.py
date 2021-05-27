import os
from dataclasses import dataclass
import unittest

import torch
from fn import _, F
from fn.iters import take

from tensorneko.util import reduce_dict_by, summarize_dict_by, generate_inf_seq, compose
from tensorneko.util.func import listdir


class UtilFuncTest(unittest.TestCase):

    def test_reduce_dict_by(self):
        x = [
            {"a": 1, "b": 2, "c": torch.Tensor([3])},
            {"a": 3, "b": 4, "c": torch.Tensor([5])},
            {"a": 2.3, "b": -1, "c": torch.Tensor([0])}
        ]
        self.assertEqual(reduce_dict_by("a", _ + _)(x), x[0]["a"] + x[1]["a"] + x[2]["a"])
        self.assertEqual(reduce_dict_by("b", _ - _)(x), x[0]["b"] - x[1]["b"] - x[2]["b"])
        self.assertEqual(reduce_dict_by("c", 10 * _ + _)(x), (x[0]["c"] * 10 + x[1]["c"]) * 10 + x[2]["c"])

    def test_summarize_dict_by(self):
        x = [
            {"a": 1, "b": torch.Tensor([2]), "c": 3},
            {"a": 3, "b": torch.Tensor([4]), "c": 5},
            {"a": 2.3, "b": torch.Tensor([-1]), "c": 0}
        ]

        self.assertEqual(summarize_dict_by("a", sum)(x), x[0]["a"] + x[1]["a"] + x[2]["a"])
        self.assertEqual(summarize_dict_by("b", torch.mean)(x), (x[0]["b"] + x[1]["b"] + x[2]["b"]) / 3)
        self.assertEqual(summarize_dict_by("c", F(map, str) >> "".join >> float)(x),
            (x[0]["c"] * 10 + x[1]["c"]) * 10 + x[2]["c"]
        )

    def test_generate_inf_seq(self):
        length = 20
        # test on single atomic types
        take_length_of = lambda l: F() >> generate_inf_seq >> F(take, l) >> list
        self.assertEqual(take_length_of(length)([3]), [3] * length)
        self.assertEqual(take_length_of(length)(["abc"]), ["abc"] * length)
        # test on multiple atomic types
        self.assertEqual(take_length_of(length * 3)([3, 4, 3]), [3, 4, 3] * length)
        self.assertEqual(take_length_of(length * 2)(["abc", 1]), ["abc", 1] * length)

        # test on reference types
        @dataclass
        class Point:
            x: float
            y: float

        seq = [{"a": 1, "b": 2}, Point(-1.2, 4.1)]
        self.assertEqual(take_length_of(length * 2)(seq), seq * length)

    def test_compose(self):
        x = 1
        for fs in [[_ + 2, _ * 2, _ ** 2, _ - 2], [str, _ + "abc", lambda s: s.replace("abc", "123")]]:
            expect = x
            for f in fs:
                expect = f(expect)

            result = compose(fs)(x)
            self.assertEqual(result, expect)

    def test_listdir(self):
        path = "."
        neko_res = listdir(path)
        os_res = os.listdir(path)
        os_res = list(map(lambda file: os.path.join(path, file), os_res))
        print(neko_res)
        for neko_path, os_path in zip(neko_res, os_res):
            self.assertEqual(neko_path, os_path)

    def test_with_printed(self):
        pass #TODO

    def test_with_printed_shape(self):
        pass #TODO

    def test_ifelse(self):
        pass #TODO
