import unittest
from typing import Union, Tuple, List

from tensorneko.util import dispatch


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
