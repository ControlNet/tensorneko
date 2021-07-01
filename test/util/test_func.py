import contextlib
import os
import unittest
from dataclasses import dataclass
from io import StringIO
from random import randint

import numpy as np
import torch
from fn import _, F
from fn.iters import take

from tensorneko.util import reduce_dict_by, summarize_dict_by, generate_inf_seq, compose
from tensorneko.util.func import listdir, with_printed, with_printed_shape, ifelse, is_bad_num


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
            {"a": 1, "b": torch.Tensor([2]), "c": 3, "d": np.array([1.5])},
            {"a": 3, "b": torch.Tensor([4]), "c": 5, "d": np.array([2.5])},
            {"a": 2.3, "b": torch.Tensor([-1]), "c": 0, "d": np.array([-1.0])}
        ]

        self.assertEqual(summarize_dict_by("a", sum)(x), x[0]["a"] + x[1]["a"] + x[2]["a"])
        self.assertEqual(summarize_dict_by("b", torch.mean)(x), (x[0]["b"] + x[1]["b"] + x[2]["b"]) / 3)
        self.assertEqual(summarize_dict_by("c", F(map, str) >> "".join >> float)(x),
            (x[0]["c"] * 10 + x[1]["c"]) * 10 + x[2]["c"]
        )
        self.assertEqual(summarize_dict_by("d", F(np.sum, axis=0))(x), x[0]["d"] + x[1]["d"] + x[2]["d"])

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
        str_io = StringIO()
        with contextlib.redirect_stdout(str_io):
            x = "abc"
            append_str = _ + "123"
            result = with_printed(x, append_str)
            output = str_io.getvalue()
        self.assertEqual(output, append_str(x) + "\n")
        self.assertEqual(result, x)

    def test_with_printed_shape(self):
        shape = torch.Size((128, 128, 32))
        x = torch.rand(shape)
        str_io = StringIO()
        with contextlib.redirect_stdout(str_io):
            result = with_printed_shape(x)
            output = str_io.getvalue()

        self.assertEqual(output, x.shape.__repr__() + "\n")
        self.assertTrue((result == x).all())

    def test_ifelse(self):
        # build functions for test
        is_even = _ % 2 == 0
        add_3 = _ + 3
        power_2 = _ ** 2

        # build composed functions for tensorneko and python3
        neko_f = ifelse(is_even, add_3, power_2)
        python_f = lambda x: x + 3 if x % 2 == 0 else x ** 2

        # compare
        for i in range(30):
            x = randint(-100, 100)
            self.assertEqual(neko_f(x), python_f(x))

    def test_is_bad_num(self):
        # test input
        inf_tensor = torch.log(torch.zeros(1, 16))
        nan_tensor = torch.zeros(1, 16) / torch.zeros(1, 16)
        # assert these are True
        self.assertTrue(is_bad_num(inf_tensor).all())
        self.assertTrue(is_bad_num(nan_tensor).all())


    def test_string_getter(self):
        pass  # TODO
