import contextlib
import os
import unittest
from dataclasses import dataclass
from io import StringIO
from random import randint

import numpy as np
import torch

from tensorneko.util import reduce_dict_by, summarize_dict_by, generate_inf_seq, compose, listdir, with_printed, \
    with_printed_shape, ifelse, is_bad_num, dict_add, tensorneko_path, as_list, identity, _, F, sparse2binary, \
    binary2sparse, circular_pad
from itertools import islice


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

        def take(n, iterable):
            return islice(iterable, n)

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

    def test_dict_add(self):
        # test merge 2 dicts
        dict_a = {"a": 1, 2: 3}
        dict_b = {"a": 2, "b": 3}
        self.assertEqual(dict_add(dict_a, dict_b), {"a": 2, 2: 3, "b": 3})
        # test merge 3 dicts
        dict_a = {"a": 1}
        dict_b = {"b": 2}
        dict_c = {"c": 3}
        self.assertEqual(dict_add(dict_a, dict_b, dict_c), {"a": 1, "b": 2, "c": 3})

    def test_tensorneko_path(self):
        # get the dir name and check if it targets to "tensorneko" directory
        if "/" in tensorneko_path:
            self.assertEqual(tensorneko_path.split("/")[-1], "tensorneko")
        elif "\\" in tensorneko_path:
            self.assertEqual(tensorneko_path.split("\\")[-1], "tensorneko")
        else:
            self.assertTrue(False)

        # test the modules are in the path
        files = os.listdir(tensorneko_path)
        self.assertTrue("callback" in files)
        self.assertTrue("io" in files)
        self.assertTrue("layer" in files)
        self.assertTrue("module" in files)
        self.assertTrue("notebook" in files)
        self.assertTrue("optim" in files)
        self.assertTrue("preprocess" in files)
        self.assertTrue("util" in files)
        self.assertTrue("visualization" in files)
        self.assertTrue("neko_model.py" in files)
        self.assertTrue("neko_module.py" in files)
        self.assertTrue("neko_trainer.py" in files)

    def test_as_list(self):
        # test list
        self.assertEqual(as_list(), [])
        self.assertEqual(as_list(1, 2, 3), [1, 2, 3])
        self.assertEqual(as_list(1, 2, 3, k1=4, k2=5, k6=111), [1, 2, 3, 4, 5, 111])

    def test_identity(self):
        # test identity
        self.assertEqual(identity(1), 1)
        self.assertEqual(identity(1, 2, 3), (1, 2, 3))
        self.assertRaises(AssertionError, identity, 1, 2, 3, 4, k=1)

    def test_sparse2binary_numpy(self):
        sparse = np.array([2, 4])
        binary = np.array([0, 0, 1, 0, 1])
        self.assertTrue(type(sparse2binary(sparse)) is np.ndarray)
        self.assertTrue((sparse2binary(sparse, 5) == binary).all())

    def test_sparse2binary_numpy_default_length(self):
        sparse = np.array([2, 5])
        binary = np.array([0, 0, 1, 0, 0, 1])
        self.assertTrue(type(sparse2binary(sparse)) is np.ndarray)
        self.assertTrue((sparse2binary(sparse) == binary).all())

    def test_sparse2binary_tensor(self):
        sparse = torch.tensor([2, 4])
        binary = torch.tensor([0, 0, 1, 0, 1])
        self.assertTrue(type(sparse2binary(sparse)) is torch.Tensor)
        self.assertTrue((sparse2binary(sparse, 5) == binary).all())

    def test_sparse2binary_tensor_default_length(self):
        sparse = torch.tensor([2, 5])
        binary = torch.tensor([0, 0, 1, 0, 0, 1])
        self.assertTrue(type(sparse2binary(sparse)) is torch.Tensor)
        self.assertTrue((sparse2binary(sparse) == binary).all())

    def test_sparse2binary_list(self):
        sparse = [2, 4]
        binary = [0, 0, 1, 0, 1]
        self.assertTrue(type(sparse2binary(sparse)) is list)
        self.assertTrue(sparse2binary(sparse, 5) == binary)

    def test_sparse2binary_list_default_length(self):
        sparse = [2, 5]
        binary = [0, 0, 1, 0, 0, 1]
        self.assertTrue(type(sparse2binary(sparse)) is list)
        self.assertTrue(sparse2binary(sparse) == binary)

    def test_binary2sparse_numpy(self):
        sparse = np.array([2, 4])
        binary = np.array([0, 0, 1, 0, 1])
        self.assertTrue(type(binary2sparse(binary)) is np.ndarray)
        self.assertTrue((binary2sparse(binary) == sparse).all())

    def test_binary2sparse_tensor(self):
        sparse = torch.tensor([2, 4])
        binary = torch.tensor([0, 0, 1, 0, 1])
        self.assertTrue(type(binary2sparse(binary)) is torch.Tensor)
        self.assertTrue((binary2sparse(binary) == sparse).all())

    def test_binary2sparse_list(self):
        sparse = [2, 4]
        binary = [0, 0, 1, 0, 1]
        self.assertTrue(type(binary2sparse(binary)) is list)
        self.assertTrue(binary2sparse(binary) == sparse)

    def test_circular_pad_list(self):
        a = [1, 2, 5]
        self.assertEqual(circular_pad(a, 0), [])
        self.assertEqual(circular_pad(a, 1), [1])
        self.assertEqual(circular_pad(a, 2), [1, 2])
        self.assertEqual(circular_pad(a, 3), [1, 2, 5])
        self.assertEqual(circular_pad(a, 4), [1, 2, 5, 1])
        self.assertEqual(circular_pad(a, 5), [1, 2, 5, 1, 2])
        self.assertEqual(circular_pad(a, 6), [1, 2, 5, 1, 2, 5])
        self.assertEqual(circular_pad(a, 7), [1, 2, 5, 1, 2, 5, 1])
        self.assertEqual(circular_pad(a, 8), [1, 2, 5, 1, 2, 5, 1, 2])
        self.assertEqual(circular_pad(a, 9), [1, 2, 5, 1, 2, 5, 1, 2, 5])
        self.assertEqual(circular_pad(a, 10), [1, 2, 5, 1, 2, 5, 1, 2, 5, 1])
