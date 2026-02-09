import unittest
import warnings
from typing import Union, Tuple, List

from tensorneko.util import dispatch
from tensorneko_util.util.dispatcher import (
    Dispatcher,
    Resolver,
    MethodResolver,
    DispatcherTypeWarning,
)


@dispatch
def f(x: int):
    return type(x)


@dispatch
def f(x: int, y: int, z: int):
    return type(x), type(y), type(z)


@dispatch
def g(x: int, y: Union[int, str]):
    return type(x), type(y)


@dispatch
def g2(x: int, y: Union[int, str] = 0):
    return type(x), type(y)


@dispatch
def h(x: int, y: str = ""):
    return type(x), type(y)


class UtilDispatcherTest(unittest.TestCase):
    @staticmethod
    def _get_register_type(func) -> List[Tuple[type, ...]]:
        return [*func._dispatcher._functions.keys()]

    def test_dispatch(self):
        registered_types = self._get_register_type(f)
        self.assertEqual(len(registered_types), 2)
        self.assertIn((int,), registered_types)
        self.assertIn((int, int, int), registered_types)
        self.assertEqual(int, f(1))
        self.assertEqual((int, int, int), f(1, 2, 3))

    def test_union_type(self):
        registered_types = self._get_register_type(g)
        self.assertEqual(len(registered_types), 2)
        self.assertIn((int, int), registered_types)
        self.assertIn((int, str), registered_types)
        self.assertEqual((int, int), g(1, 2))
        self.assertEqual((int, str), g(1, "2"))

    def test_default_type(self):
        registered_types = self._get_register_type(h)
        self.assertEqual(len(registered_types), 2)
        self.assertIn((int,), registered_types)
        self.assertIn((int, str), registered_types)
        self.assertEqual((int, str), h(1))
        self.assertEqual((int, str), h(1, "2"))

    def test_union_with_default(self):
        registered_types = self._get_register_type(g2)
        self.assertEqual(len(registered_types), 3)
        self.assertIn((int,), registered_types)
        self.assertIn((int, int), registered_types)
        self.assertIn((int, str), registered_types)
        self.assertEqual((int, int), g2(1))
        self.assertEqual((int, int), g2(1, 2))
        self.assertEqual((int, str), g2(1, "2"))

    def test_dispatch_invalid_type_raises(self):
        """Calling dispatched function with unregistered types raises TypeError."""
        with self.assertRaises(TypeError):
            f(1.5)  # float not registered for f

    def test_dispatch_of_with_explicit_types(self):
        """dispatch.of() with explicit types registers a function."""

        @dispatch.of(float, float)
        def add_floats(x, y):
            return x + y

        result = add_floats(1.0, 2.0)
        self.assertEqual(result, 3.0)

    def test_dispatch_of_with_base(self):
        """dispatch.of() with base argument uses existing dispatcher."""

        @dispatch
        def myfunc(x: int) -> str:
            return "int"

        @dispatch.of(str, base=myfunc)
        def myfunc_str(x):
            return "str"

        self.assertEqual(myfunc(1), "int")
        self.assertEqual(myfunc("hello"), "str")

    def test_dispatch_no_annotation_warns(self):
        """dispatch on function without annotations should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @dispatch
            def no_ann(x):
                return x

            type_warns = [x for x in w if issubclass(x.category, DispatcherTypeWarning)]
            self.assertGreater(len(type_warns), 0)

    def test_dispatch_duplicate_type_warns(self):
        """Line 92: registering duplicate type signature warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @dispatch
            def dup_func(x: int) -> int:
                return x

            @dispatch
            def dup_func(x: int) -> int:
                return x * 2

            override_warns = [
                x
                for x in w
                if issubclass(x.category, DispatcherTypeWarning)
                and "overridden" in str(x.message)
            ]
            self.assertGreater(len(override_warns), 0)

    def test_dispatch_staticmethod_unwrap(self):
        """Line 39: staticmethod is unwrapped by Dispatcher.__call__."""
        d = Dispatcher("test_static_unwrap")

        @staticmethod
        def static_func(x: int) -> int:
            return x + 1

        resolver = d(static_func)
        self.assertEqual(resolver(5), 6)
