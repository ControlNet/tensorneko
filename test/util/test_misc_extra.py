import contextlib
import os
import tempfile
import unittest
from io import StringIO
from itertools import islice
from unittest.mock import patch

from tensorneko_util.util import (
    as_list,
    compose,
    dict_add,
    generate_inf_seq,
    identity,
    ifelse,
    list_to_dict,
    listdir,
    load_py,
    sample_indexes,
    try_until_success,
    with_printed,
)


class MiscExtraTest(unittest.TestCase):
    def test_generate_inf_seq_repeats_and_keeps_reference(self):
        self.assertEqual(list(islice(generate_inf_seq([1, 2]), 6)), [1, 2, 1, 2, 1, 2])

        obj = {"k": "v"}
        seq = list(islice(generate_inf_seq([obj]), 3))
        self.assertIs(seq[0], obj)
        self.assertIs(seq[1], obj)
        self.assertIs(seq[2], obj)

    def test_compose_listdir_and_with_printed(self):
        plus_one_then_str = compose([lambda x: x + 1, str])
        self.assertEqual(plus_one_then_str(41), "42")

        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, "a.txt"), "w").close()
            open(os.path.join(temp_dir, "b.py"), "w").close()
            listed = listdir(temp_dir, lambda name: name.endswith(".py"))
            self.assertEqual(listed, [os.path.join(temp_dir, "b.py")])

        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            value = with_printed("hello")
        self.assertEqual(stream.getvalue(), "hello\n")
        self.assertEqual(value, "hello")

    def test_ifelse_dict_add_as_list_identity_and_list_to_dict(self):
        f = ifelse(lambda x: x > 0, lambda x: x * 2)
        self.assertEqual(f(3), 6)
        self.assertEqual(f(-5), -5)

        self.assertEqual(dict_add({"a": 1}, {"a": 2, "b": 3}), {"a": 2, "b": 3})
        self.assertEqual(as_list(1, 2, x=3, y=4), [1, 2, 3, 4])
        self.assertEqual(identity(10), 10)
        self.assertEqual(identity(1, 2), (1, 2))
        with self.assertRaises(TypeError):
            identity(1, kw=2)

        records = ["aa", "bb"]
        self.assertEqual(list_to_dict(records, key=len), {2: "bb"})

    def test_load_py(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = os.path.join(temp_dir, "temp_mod.py")
            with open(module_path, "w") as f:
                f.write("VALUE = 7\n")
                f.write("def add_one(x):\n")
                f.write("    return x + 1\n")

            module = load_py(module_path)
            self.assertEqual(module.VALUE, 7)
            self.assertEqual(module.add_one(2), 3)

    def test_try_until_success_success_and_raise(self):
        callback_errors = []
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("retry")
            return "ok"

        result = try_until_success(
            flaky,
            max_trials=5,
            sleep_time=0.01,
            exception_callback=lambda e: callback_errors.append(str(e)),
            exception_type=ValueError,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(callback_errors, ["retry", "retry"])

        fail_count = {"n": 0}

        def always_fail():
            fail_count["n"] += 1
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            try_until_success(always_fail, max_trials=2, exception_type=RuntimeError)
        self.assertEqual(fail_count["n"], 3)

    def test_sample_indexes_with_patched_start(self):
        with patch("tensorneko_util.util.misc.np.random.randint", return_value=2):
            indexes = sample_indexes(total_frames=10, n_frames=3, sample_rate=2)
        self.assertEqual(indexes.tolist(), [2, 4, 6])
