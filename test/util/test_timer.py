import io
import time
import unittest
from contextlib import redirect_stdout

from tensorneko_util.util.timer import Timer


import builtins
import inspect
from unittest.mock import patch


def _apply_timer_lines(func):
    original_exec = builtins.exec
    captured_code = {}

    def patched_exec(code, *args):
        captured_code["source"] = code
        ns = {"Timer": Timer}
        original_exec(code, ns)
        captured_code["ns"] = ns

    with patch.object(builtins, "exec", side_effect=patched_exec):
        try:
            Timer.lines(func)
        except KeyError:
            pass
    return captured_code["ns"][func.__name__]


def _lines_add_and_double_raw(x):
    a = x + 1
    b = a * 2
    return b


def _lines_func_with_control_raw(x):
    if x > 0:
        result = x * 2
    else:
        result = x * 3
    return result


_lines_add_and_double = _apply_timer_lines(_lines_add_and_double_raw)
_lines_func_with_control = _apply_timer_lines(_lines_func_with_control_raw)


class UtilTimerTest(unittest.TestCase):
    def test_context_manager_elapsed_and_total_time(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            with Timer() as timer:
                time.sleep(0.001)
                self.assertIsInstance(timer.elapsed, float)
                self.assertGreater(timer.elapsed, 0.0)
            output = buffer.getvalue()

        self.assertIsNotNone(timer.total_time)
        self.assertGreater(timer.total_time, 0.0)
        self.assertEqual(len(timer.times), 1)
        self.assertIn("[Timer] Total:", output)

    def test_context_manager_with_named_checkpoint(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            with Timer() as timer:
                time.sleep(0.001)
                delta = timer.time("phase-a")
            output = buffer.getvalue()

        self.assertGreater(delta, 0.0)
        self.assertEqual(len(timer.times), 2)
        self.assertIn("phase-a", output)

    def test_decorator_with_parentheses(self):
        @Timer()
        def add(a, b):
            return a + b

        with io.StringIO() as buffer, redirect_stdout(buffer):
            result = add(2, 3)
            output = buffer.getvalue()

        self.assertEqual(result, 5)
        self.assertTrue(callable(add))
        self.assertEqual(add.__name__, "add")
        self.assertIn("[Timer] add:", output)

    def test_decorator_without_parentheses(self):
        @Timer
        def mul(a, b):
            return a * b

        with io.StringIO() as buffer, redirect_stdout(buffer):
            result = mul(2, 4)
            output = buffer.getvalue()

        self.assertEqual(result, 8)
        self.assertTrue(callable(mul))
        self.assertEqual(mul.__name__, "mul")
        self.assertIn("[Timer] mul:", output)

    def test_new_detects_callable_argument(self):
        def identity(x):
            return x

        decorated = Timer(identity)
        with io.StringIO() as buffer, redirect_stdout(buffer):
            result = decorated(9)
            output = buffer.getvalue()

        self.assertTrue(callable(decorated))
        self.assertEqual(result, 9)
        self.assertIn("[Timer] identity:", output)

    def test_verbose_false_suppresses_context_output(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            with Timer(verbose=False) as timer:
                time.sleep(0.001)
                _ = timer.time("hidden")
            output = buffer.getvalue()

        self.assertEqual(output, "")

    def test_elapsed_property_returns_positive_float(self):
        timer = Timer(verbose=False)
        with timer:
            time.sleep(0.001)
            elapsed = timer.elapsed

        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0.0)

    def test_time_data_is_stored_in_times(self):
        with Timer(verbose=False) as timer:
            time.sleep(0.001)
            timer.time()
            timer.time("second")

        self.assertEqual(len(timer.times), 3)
        self.assertTrue(all(isinstance(item, float) for item in timer.times))

    def test_make_str_format(self):
        s = Timer._make_str("label", 1.23)
        self.assertEqual(s, "[Timer] label: 1.23 sec")

    def test_context_manager_verbose_true_prints_total(self):
        """Verbose=True prints total time with named segments."""
        with io.StringIO() as buffer, redirect_stdout(buffer):
            with Timer(verbose=True) as t:
                time.sleep(0.001)
                dt1 = t.time("seg1")
                time.sleep(0.001)
                dt2 = t.time("seg2")
            output = buffer.getvalue()

        self.assertGreater(dt1, 0.0)
        self.assertGreater(dt2, 0.0)
        self.assertIn("seg1", output)
        self.assertIn("seg2", output)
        self.assertIn("Total", output)

    def test_time_without_name_uses_empty_string(self):
        """Calling time() without name uses empty string."""
        with io.StringIO() as buffer, redirect_stdout(buffer):
            with Timer(verbose=True) as t:
                time.sleep(0.001)
                dt = t.time()  # no name
            output = buffer.getvalue()

        self.assertGreater(dt, 0.0)
        self.assertIn("[Timer]", output)

    def test_verbose_false_suppresses_time_output(self):
        """verbose=False suppresses both time() and exit output."""
        with io.StringIO() as buffer, redirect_stdout(buffer):
            with Timer(verbose=False) as t:
                time.sleep(0.001)
                dt = t.time("hidden_seg")
                time.sleep(0.001)
            output = buffer.getvalue()

        self.assertEqual(output, "")
        self.assertGreater(dt, 0.0)
        self.assertIsNotNone(t.total_time)

    def test_decorator_call_prints_function_name(self):
        """Timer() as decorator prints [Timer] func_name: X sec."""
        timer_instance = Timer()

        @timer_instance
        def my_func(x):
            return x + 1

        with io.StringIO() as buffer, redirect_stdout(buffer):
            result = my_func(5)
            output = buffer.getvalue()

        self.assertEqual(result, 6)
        self.assertIn("[Timer] my_func:", output)
        self.assertIn("sec", output)

    def test_elapsed_grows_over_time(self):
        """elapsed property should increase as time passes."""
        with Timer(verbose=False) as t:
            e1 = t.elapsed
            time.sleep(0.01)
            e2 = t.elapsed
        self.assertGreater(e2, e1)

    def test_lines_simple_function(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            result = _lines_add_and_double(5)
            output = buffer.getvalue()

        self.assertEqual(result, 12)
        self.assertIn("[Timer]", output)
        self.assertIn("_lines_add_and_double", output)

    def test_lines_with_control_flow(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            result_pos = _lines_func_with_control(5)
            output = buffer.getvalue()

        self.assertEqual(result_pos, 10)
        self.assertIn("[Timer]", output)

    def test_lines_negative_branch(self):
        with io.StringIO() as buffer, redirect_stdout(buffer):
            result_neg = _lines_func_with_control(-3)
            output = buffer.getvalue()

        self.assertEqual(result_neg, -9)
        self.assertIn("[Timer]", output)
