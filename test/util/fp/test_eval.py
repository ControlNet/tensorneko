import unittest

from tensorneko_util.util.fp.monad.eval import Eval, Always, Later, Now


class TestEvalAlways(unittest.TestCase):
    """Test Eval.always — call-by-name: re-evaluates every access."""

    def test_always_returns_always_instance(self):
        e = Eval.always(lambda: 42)
        self.assertIsInstance(e, Always)

    def test_always_value(self):
        e = Eval.always(lambda: 10)
        self.assertEqual(e.value, 10)

    def test_always_re_evaluates(self):
        counter = [0]

        def inc():
            counter[0] += 1
            return counter[0]

        e = Eval.always(inc)
        self.assertEqual(e.value, 1)
        self.assertEqual(e.value, 2)
        self.assertEqual(e.value, 3)

    def test_always_evaluated_property(self):
        e = Eval.always(lambda: 1)
        # Always never caches, so evaluated is always False before access
        self.assertFalse(e.evaluated)
        _ = e.value
        # Still False because Always doesn't store _value
        self.assertFalse(e.evaluated)

    def test_always_map(self):
        e = Eval.always(lambda: 5)
        mapped = e.map(lambda x: x * 2)
        self.assertIsInstance(mapped, Always)
        self.assertEqual(mapped.value, 10)

    def test_always_flat_map(self):
        e = Eval.always(lambda: 5)
        result = e.flat_map(lambda x: Eval.always(lambda: x + 1))
        self.assertEqual(result.value, 6)

    def test_always_str_repr(self):
        e = Eval.always(lambda: 42)
        s = str(e)
        self.assertIn("Always", s)
        r = repr(e)
        self.assertEqual(s, r)


class TestEvalLater(unittest.TestCase):
    """Test Eval.later — call-by-need: evaluates once, caches."""

    def test_later_returns_later_instance(self):
        e = Eval.later(lambda: 42)
        self.assertIsInstance(e, Later)

    def test_later_value(self):
        e = Eval.later(lambda: 99)
        self.assertEqual(e.value, 99)

    def test_later_evaluates_once(self):
        counter = [0]

        def inc():
            counter[0] += 1
            return counter[0]

        e = Eval.later(inc)
        self.assertEqual(e.value, 1)
        self.assertEqual(e.value, 1)  # cached
        self.assertEqual(e.value, 1)  # still cached

    def test_later_evaluated_property(self):
        e = Eval.later(lambda: 1)
        self.assertFalse(e.evaluated)
        _ = e.value
        self.assertTrue(e.evaluated)

    def test_later_map(self):
        e = Eval.later(lambda: 3)
        mapped = e.map(lambda x: x**2)
        self.assertEqual(mapped.value, 9)


class TestEvalNow(unittest.TestCase):
    """Test Eval.now — call-by-value: evaluates immediately."""

    def test_now_returns_now_instance(self):
        e = Eval.now(lambda: 42)
        self.assertIsInstance(e, Now)

    def test_now_evaluates_immediately(self):
        counter = [0]

        def inc():
            counter[0] += 1
            return counter[0]

        e = Eval.now(inc)
        self.assertEqual(counter[0], 1)  # evaluated during __init__
        self.assertEqual(e.value, 1)

    def test_now_value(self):
        e = Eval.now(lambda: 77)
        self.assertEqual(e.value, 77)

    def test_now_evaluated_property(self):
        e = Eval.now(lambda: 1)
        self.assertTrue(e.evaluated)


class TestEvalPure(unittest.TestCase):
    def test_pure_is_abstract(self):
        """Eval.pure tries to instantiate Eval (abstract) — should raise TypeError."""
        with self.assertRaises(TypeError):
            Eval.pure(42)


class TestEvalFlatten(unittest.TestCase):
    def test_flatten_later(self):
        inner = Eval.later(lambda: 5)
        outer = Eval.later(lambda: inner)
        result = outer.flatten()
        self.assertEqual(result.value, 5)


class TestEvalDecorator(unittest.TestCase):
    """Test Eval.always, Eval.later, Eval.now as decorators."""

    def test_always_as_decorator(self):
        @Eval.always
        def call_by_name_var():
            return 42

        self.assertIsInstance(call_by_name_var, Always)
        self.assertEqual(call_by_name_var.value, 42)

    def test_later_as_decorator(self):
        @Eval.later
        def call_by_need_var():
            return 43

        self.assertIsInstance(call_by_need_var, Later)
        self.assertEqual(call_by_need_var.value, 43)

    def test_now_as_decorator(self):
        @Eval.now
        def call_by_value_var():
            return 44

        self.assertIsInstance(call_by_value_var, Now)
        self.assertEqual(call_by_value_var.value, 44)


if __name__ == "__main__":
    unittest.main()
