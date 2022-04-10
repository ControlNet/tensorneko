import operator
import unittest

from tensorneko_util.util import curry, F, _


class UtilCurryTest(unittest.TestCase):

    def _assert_instance(self, expected, actual):
        self.assertEqual(expected.__module__, actual.__module__)
        self.assertEqual(expected.__name__, actual.__name__)

    def test_curry_wrapper(self):

        @curry
        def _child(a, b, c, d):
            return a + b + c + d

        @curry
        def _moma(a, b):
            return _child(a, b)

        res1 = _moma(1)
        self._assert_instance(_moma, res1)
        res2 = res1(2)
        self._assert_instance(_child, res2)
        res3 = res2(3)
        self._assert_instance(_child, res3)
        res4 = res3(4)

        self.assertEqual(res4, 10)

    def test_curried_with_annotations_when_they_are_supported(self):

        def _custom_sum(a, b, c, d):
            return a + b + c + d

        _custom_sum.__annotations__ = {
                                        'a': int,
                                        'b': int,
                                        'c': int,
                                        'd': int,
                                        'return': int
                                      }

        custom_sum = curry(_custom_sum)

        res1 = custom_sum(1)
        self._assert_instance(custom_sum, res1)
        res2 = res1(2)
        self._assert_instance(custom_sum, res2)
        res3 = res2(3)
        self._assert_instance(custom_sum, res3)
        res4 = res3(4)

        self.assertEqual(res4, 10)


class UtilFuncTest(unittest.TestCase):

    def test_composition(self):
        def f(x):
            return x * 2

        def g(x):
            return x + 10

        self.assertEqual(30, (F(f) << g)(5))

        def z(x):
            return x * 20
        self.assertEqual(220, (F(f) << F(g) << F(z))(5))

    def test_partial(self):
        # Partial should work if we pass additional arguments to F constructor
        f = F(operator.add, 10) << F(operator.add, 5)
        self.assertEqual(25, f(10))

    def test_underscore(self):
        self.assertEqual(
            [1, 4, 9],
            list(map(F() << (_ ** 2) << _ + 1, range(3)))
        )

    def test_pipe_composition(self):
        def f(x):
            return x * 2

        def g(x):
            return x + 10

        self.assertEqual(20, (F() >> f >> g)(5))

    def test_pipe_partial(self):
        func = F() >> (filter, _ < 6) >> sum
        self.assertEqual(15, func(range(10)))
