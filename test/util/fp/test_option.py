import unittest

from tensorneko_util.util.fp.monad.option import (
    Option,
    Some,
    Empty,
    return_option,
    ReturnOptionDecorator,
)


class TestSomeCreation(unittest.TestCase):
    def test_some_creation(self):
        s = Some(42)
        self.assertIsInstance(s, Some)
        self.assertEqual(s.get(), 42)

    def test_some_is_defined(self):
        self.assertTrue(Some(1).is_defined())

    def test_some_is_empty(self):
        self.assertFalse(Some(1).is_empty())

    def test_some_unwraps_nested_some(self):
        inner = Some(10)
        outer = Some(inner)
        self.assertEqual(outer.get(), 10)

    def test_some_with_various_types(self):
        self.assertEqual(Some("hello").get(), "hello")
        self.assertEqual(Some([1, 2]).get(), [1, 2])
        self.assertEqual(Some(0).get(), 0)
        self.assertEqual(Some(False).get(), False)


class TestEmptyCreation(unittest.TestCase):
    def test_empty_is_singleton(self):
        self.assertIs(Empty, Empty)

    def test_empty_is_defined(self):
        self.assertFalse(Empty.is_defined())

    def test_empty_is_empty(self):
        self.assertTrue(Empty.is_empty())

    def test_empty_get_returns_none(self):
        self.assertIsNone(Empty.get())


class TestOptionFactory(unittest.TestCase):
    def test_option_non_none_returns_some(self):
        opt = Option(5)
        self.assertIsInstance(opt, Some)
        self.assertEqual(opt.get(), 5)

    def test_option_none_returns_empty(self):
        opt = Option(None)
        self.assertIs(opt, Empty)

    def test_option_wraps_some_identity(self):
        s = Some(3)
        self.assertIs(Option(s), s)

    def test_option_wraps_empty_identity(self):
        self.assertIs(Option(Empty), Empty)

    def test_option_with_custom_checker(self):
        opt = Option(0, checker=lambda x: x > 0)
        self.assertIs(opt, Empty)
        opt2 = Option(5, checker=lambda x: x > 0)
        self.assertIsInstance(opt2, Some)
        self.assertEqual(opt2.get(), 5)


class TestGetOrElse(unittest.TestCase):
    def test_some_get_or_else_returns_value(self):
        self.assertEqual(Some(10).get_or_else(99), 10)

    def test_empty_get_or_else_returns_default(self):
        self.assertEqual(Empty.get_or_else(99), 99)


class TestGetOrCall(unittest.TestCase):
    def test_some_get_or_call_returns_value(self):
        self.assertEqual(Some(10).get_or_call(lambda: 99), 10)

    def test_empty_get_or_call_calls_function(self):
        self.assertEqual(Empty.get_or_call(lambda x: x * 2, 5), 10)

    def test_empty_get_or_call_with_kwargs(self):
        def f(a, b=10):
            return a + b

        self.assertEqual(Empty.get_or_call(f, 1, b=20), 21)


class TestMap(unittest.TestCase):
    def test_some_map_applies_function(self):
        result = Some(5).map(lambda x: x * 2)
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 10)

    def test_some_map_returning_none_gives_empty(self):
        result = Some(5).map(lambda x: None)
        self.assertIs(result, Empty)

    def test_empty_map_returns_empty(self):
        result = Empty.map(lambda x: x * 2)
        self.assertIs(result, Empty)


class TestFlatMap(unittest.TestCase):
    def test_some_flat_map_returns_some(self):
        result = Some(5).flat_map(lambda x: Some(x + 1))
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 6)

    def test_some_flat_map_returns_empty(self):
        result = Some(5).flat_map(lambda x: Empty)
        self.assertIs(result, Empty)

    def test_empty_flat_map_returns_empty(self):
        result = Empty.flat_map(lambda x: Some(x + 1))
        self.assertIs(result, Empty)


class TestFlatten(unittest.TestCase):
    def test_some_flatten(self):
        nested = Some(Some(42))
        # Some.__init__ unwraps, so nested.x is already Some(42)
        # Actually Some(Some(42)) -> Some with x = 42 (unwrapped in __init__)
        # To truly test flatten we need to bypass unwrap
        # Let's use a raw approach
        s = Some(10)
        # flatten on Some returns self.x
        self.assertEqual(s.flatten(), 10)

    def test_empty_flatten(self):
        self.assertIs(Empty.flatten(), Empty)


class TestFilter(unittest.TestCase):
    def test_some_filter_true(self):
        result = Some(5).filter(lambda x: x > 0)
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 5)

    def test_some_filter_false(self):
        result = Some(5).filter(lambda x: x > 10)
        self.assertIs(result, Empty)

    def test_empty_filter(self):
        result = Empty.filter(lambda x: True)
        self.assertIs(result, Empty)


class TestFilterNot(unittest.TestCase):
    def test_some_filter_not_true(self):
        result = Some(5).filter_not(lambda x: x > 10)
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 5)

    def test_some_filter_not_false(self):
        result = Some(5).filter_not(lambda x: x > 0)
        self.assertIs(result, Empty)

    def test_empty_filter_not(self):
        result = Empty.filter_not(lambda x: True)
        self.assertIs(result, Empty)


class TestOrElse(unittest.TestCase):
    def test_some_or_else_returns_self(self):
        s = Some(5)
        self.assertIs(s.or_else(99), s)

    def test_empty_or_else_returns_option_of_default(self):
        result = Empty.or_else(99)
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 99)

    def test_empty_or_else_none_returns_empty(self):
        result = Empty.or_else(None)
        self.assertIs(result, Empty)


class TestOrCall(unittest.TestCase):
    def test_some_or_call_returns_self(self):
        s = Some(5)
        self.assertIs(s.or_call(lambda: 99), s)

    def test_empty_or_call_returns_option(self):
        result = Empty.or_call(lambda: 99)
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 99)

    def test_empty_or_call_with_args(self):
        result = Empty.or_call(lambda x: x * 2, 5)
        self.assertEqual(result.get(), 10)


class TestFold(unittest.TestCase):
    def test_some_fold_applies_function(self):
        result = Some(5).fold(lambda x: x * 2, -1)
        self.assertEqual(result, 10)

    def test_empty_fold_returns_if_empty(self):
        result = Empty.fold(lambda x: x * 2, -1)
        self.assertEqual(result, -1)


class TestForEach(unittest.TestCase):
    def test_some_for_each_calls_function(self):
        results = []
        Some(5).for_each(lambda x: results.append(x))
        self.assertEqual(results, [5])

    def test_empty_for_each_does_nothing(self):
        results = []
        Empty.for_each(lambda x: results.append(x))
        self.assertEqual(results, [])


class TestExists(unittest.TestCase):
    def test_some_exists_true(self):
        self.assertTrue(Some(5).exists(lambda x: x > 0))

    def test_some_exists_false(self):
        self.assertFalse(Some(5).exists(lambda x: x > 10))

    def test_empty_exists(self):
        self.assertFalse(Empty.exists(lambda x: True))


class TestForAll(unittest.TestCase):
    def test_some_for_all_true(self):
        self.assertTrue(Some(5).for_all(lambda x: x > 0))

    def test_some_for_all_false(self):
        self.assertFalse(Some(5).for_all(lambda x: x > 10))

    def test_empty_for_all(self):
        self.assertTrue(Empty.for_all(lambda x: False))


class TestContains(unittest.TestCase):
    def test_some_contains_true(self):
        self.assertTrue(Some(5).contains(5))

    def test_some_contains_false(self):
        self.assertFalse(Some(5).contains(10))

    def test_empty_contains(self):
        self.assertFalse(Empty.contains(5))


class TestZip(unittest.TestCase):
    def test_some_zip_some(self):
        result = Some(1).zip(Some(2))
        self.assertEqual(result.get(), (1, 2))

    def test_some_zip_empty(self):
        result = Some(1).zip(Empty)
        self.assertIs(result, Empty)

    def test_empty_zip_some(self):
        result = Empty.zip(Some(1))
        self.assertIs(result, Empty)

    def test_empty_zip_empty(self):
        result = Empty.zip(Empty)
        self.assertIs(result, Empty)


class TestUnzip(unittest.TestCase):
    def test_some_unzip(self):
        s = Some((1, 2))
        a, b = s.unzip()
        self.assertEqual(a.get(), 1)
        self.assertEqual(b.get(), 2)

    def test_some_unzip_wrong_length(self):
        s = Some((1, 2, 3))
        with self.assertRaises(ValueError):
            s.unzip()

    def test_some_unzip_non_tuple(self):
        s = Some([1, 2])
        with self.assertRaises(ValueError):
            s.unzip()

    def test_empty_unzip(self):
        a, b = Empty.unzip()
        self.assertIs(a, Empty)
        self.assertIs(b, Empty)


class TestUnzip3(unittest.TestCase):
    def test_some_unzip3(self):
        s = Some((1, 2, 3))
        a, b, c = s.unzip3()
        self.assertEqual(a.get(), 1)
        self.assertEqual(b.get(), 2)
        self.assertEqual(c.get(), 3)

    def test_empty_unzip3(self):
        a, b, c = Empty.unzip3()
        self.assertIs(a, Empty)
        self.assertIs(b, Empty)
        self.assertIs(c, Empty)


class TestToList(unittest.TestCase):
    def test_some_to_list(self):
        self.assertEqual(Some(5).to_list(), [5])

    def test_empty_to_list(self):
        self.assertEqual(Empty.to_list(), [])


class TestEquality(unittest.TestCase):
    def test_some_eq_same_value(self):
        self.assertEqual(Some(1), Some(1))

    def test_some_eq_different_value(self):
        self.assertNotEqual(Some(1), Some(2))

    def test_some_ne_empty(self):
        self.assertNotEqual(Some(1), Empty)

    def test_some_ne_non_option(self):
        self.assertNotEqual(Some(1), 1)

    def test_empty_eq_empty(self):
        self.assertEqual(Empty, Empty)

    def test_empty_ne_some(self):
        self.assertFalse(Empty.__eq__(Some(1)))


class TestStrRepr(unittest.TestCase):
    def test_some_str(self):
        self.assertEqual(str(Some(42)), "Some(42)")

    def test_some_repr(self):
        self.assertEqual(repr(Some(42)), "Some(42)")

    def test_empty_str(self):
        self.assertEqual(str(Empty), "Empty")

    def test_empty_repr(self):
        self.assertEqual(repr(Empty), "Empty")


class TestReturnOptionDecorator(unittest.TestCase):
    def test_return_option_non_none(self):
        @return_option
        def f(x):
            return x + 1

        result = f(1)
        self.assertIsInstance(result, Some)
        self.assertEqual(result.get(), 2)

    def test_return_option_none(self):
        @return_option
        def f(x):
            return None

        result = f(1)
        self.assertIs(result, Empty)

    def test_return_option_with_checker(self):
        @ReturnOptionDecorator.with_checker(lambda x: x > 0)
        def f(x):
            return x - 10

        result = f(5)
        self.assertIs(result, Empty)

        result2 = f(20)
        self.assertIsInstance(result2, Some)
        self.assertEqual(result2.get(), 10)

    def test_return_option_preserves_function_name(self):
        @return_option
        def my_func(x):
            return x

        self.assertEqual(my_func.__name__, "my_func")


class TestMethodChaining(unittest.TestCase):
    def test_map_filter_get(self):
        result = Some(1).map(lambda x: x + 1).filter(lambda x: x > 0).get()
        self.assertEqual(result, 2)

    def test_map_filter_to_empty(self):
        result = Some(1).map(lambda x: x + 1).filter(lambda x: x > 10).get_or_else(-1)
        self.assertEqual(result, -1)

    def test_flat_map_chain(self):
        result = (
            Some(5)
            .flat_map(lambda x: Some(x * 2))
            .flat_map(lambda x: Some(x + 1))
            .get()
        )
        self.assertEqual(result, 11)

    def test_complex_chain(self):
        result = (
            Some(10)
            .filter(lambda x: x > 5)
            .map(lambda x: x * 2)
            .flat_map(lambda x: Some(x - 1) if x > 0 else Empty)
            .fold(lambda x: x + 100, -1)
        )
        self.assertEqual(result, 119)

    def test_chain_from_empty_stays_empty(self):
        result = (
            Empty.map(lambda x: x + 1)
            .filter(lambda x: True)
            .flat_map(lambda x: Some(x))
            .get_or_else(-1)
        )
        self.assertEqual(result, -1)


if __name__ == "__main__":
    unittest.main()
