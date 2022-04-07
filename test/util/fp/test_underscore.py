import unittest
from functools import reduce

from tensorneko_util.util.fp import _, underscore


class UtilUnderscoreTest(unittest.TestCase):
    def test_identity_default(self):
        self.assertEqual(10, _(10))

    def test_arithmetic(self):
        # operator +
        self.assertEqual(7, (_ + 2)(5))
        self.assertEqual([10, 11, 12], list(map(_ + 10, [0, 1, 2])))
        # operator -
        self.assertEqual(3, (_ - 2)(5))
        self.assertEqual(13, (_ - 2 + 10)(5))
        self.assertEqual([0, 1, 2], list(map(_ - 10, [10, 11, 12])))
        # operator *
        self.assertEqual(10, (_ * 2)(5))
        self.assertEqual(50, (_ * 2 + 40)(5))
        self.assertEqual([0, 10, 20], list(map(_ * 10, [0, 1, 2])))
        # operator /
        self.assertEqual(5, (_ / 2)(10))
        self.assertEqual(6, (_ / 2 + 1)(10))
        self.assertEqual([1, 2, 3], list(map(_ / 10, [10, 20, 30])))
        # operator **
        self.assertEqual(100, (_ ** 2)(10))
        # operator %
        self.assertEqual(1, (_ % 2)(11))
        # operator <<
        self.assertEqual(32, (_ << 2)(8))
        # operator >>
        self.assertEqual(2, (_ >> 2)(8))
        # operator (-a)
        self.assertEqual(10,  (-_)(-10))
        self.assertEqual(-10, (-_)(10))
        # operator (+a)
        self.assertEqual(10,  (+_)(10))
        self.assertEqual(-10, (+_)(-10))
        # operator (~a)
        self.assertEqual(-11, (~_)(10))

    def test_arithmetic_multiple(self):
        self.assertEqual(10, (_ + _)(5, 5))
        self.assertEqual(0, (_ - _)(5, 5))
        self.assertEqual(25, (_ * _)(5, 5))
        self.assertEqual(1, (_ / _)(5, 5))

    def test_arithmetic_swap(self):
        # operator +
        self.assertEqual(7, (2 + _)(5))
        self.assertEqual([10, 11, 12], list(map(10 + _, [0, 1, 2])))
        # operator -
        self.assertEqual(3, (8 - _)(5))
        self.assertEqual(13, (8 - _ + 10)(5))
        self.assertEqual([10, 9, 8], list(map(10 - _, [0, 1, 2])))
        # operator *
        self.assertEqual(10, (2 * _)(5))
        self.assertEqual(50, (2 * _ + 40)(5))
        self.assertEqual([0, 10, 20], list(map(10 * _, [0, 1, 2])))
        # operator /
        self.assertEqual(5, (10 / _)(2))
        self.assertEqual(6, (10 / _ + 1)(2))
        self.assertEqual([10, 5, 2], list(map(100 / _, [10, 20, 50])))
        # operator **
        self.assertEqual(100, (10**_)(2))
        # operator %
        self.assertEqual(1, (11 % _)(2))
        # operator <<
        self.assertEqual(32, (8 << _)(2))
        # operator >>
        self.assertEqual(2, (8 >> _)(2))

    def test_bitwise(self):
        # and
        self.assertTrue((_ & 1)(1))
        self.assertFalse((_ & 1)(0))
        self.assertFalse((_ & 0)(1))
        self.assertFalse((_ & 0)(0))
        # or
        self.assertTrue((_ | 1)(1))
        self.assertTrue((_ | 1)(0))
        self.assertTrue((_ | 0)(1))
        self.assertFalse((_ | 0)(0))
        # xor
        self.assertTrue((_ ^ 1)(0))
        self.assertTrue((_ ^ 0)(1))
        self.assertFalse((_ ^ 1)(1))
        self.assertFalse((_ ^ 0)(0))

    def test_bitwise_swap(self):
        # and
        self.assertTrue((1 & _)(1))
        self.assertFalse((1 & _)(0))
        self.assertFalse((0 & _)(1))
        self.assertFalse((0 & _)(0))
        # or
        self.assertTrue((1 | _)(1))
        self.assertTrue((1 | _)(0))
        self.assertTrue((0 | _)(1))
        self.assertFalse((0 | _)(0))
        # xor
        self.assertTrue((1 ^ _)(0))
        self.assertTrue((0 ^ _)(1))
        self.assertFalse((1 ^ _)(1))
        self.assertFalse((0 ^ _)(0))

    def test_getattr(self):
        class GetattrTest(object):
            def __init__(self):
                self.doc = "TestCase"

        self.assertEqual("TestCase", (_.doc)(GetattrTest()))
        self.assertEqual("TestCaseTestCase", (_.doc * 2)(GetattrTest()))
        self.assertEqual(
            "TestCaseTestCase", (_.doc + _.doc)(GetattrTest(), GetattrTest())
        )

    def test_call_method(self):
        self.assertEqual(["test", "case"], (_.call("split"))("test case"))
        self.assertEqual("str", _.__name__(str))

    def test_call_method_args(self):
        self.assertEqual(["test", "case"], (_.call("split", "-"))("test-case"))
        self.assertEqual(["test-case"], (_.call("split", "-", 0))("test-case"))

    def test_call_method_kwargs(self):
        test_dict = {'num': 23}
        _.call("update", num=42)(test_dict)
        self.assertEqual({'num': 42}, (test_dict))

    def test_comparator(self):
        self.assertTrue((_ < 7)(1))
        self.assertFalse((_ < 7)(10))
        self.assertTrue((_ > 20)(25))
        self.assertFalse((_ > 20)(0))
        self.assertTrue((_ <= 7)(6))
        self.assertTrue((_ <= 7)(7))
        self.assertFalse((_ <= 7)(8))
        self.assertTrue((_ >= 7)(8))
        self.assertTrue((_ >= 7)(7))
        self.assertFalse((_ >= 7)(6))
        self.assertTrue((_ == 10)(10))
        self.assertFalse((_ == 10)(9))

    def test_none(self):
        self.assertTrue((_ == None)(None))  # noqa: E711

        class pushlist(list):
            def __lshift__(self, item):
                self.append(item)
                return self

        self.assertEqual([None], (_ << None)(pushlist()))

    def test_comparator_multiple(self):
        self.assertTrue((_ < _)(1, 2))
        self.assertFalse((_ < _)(2, 1))
        self.assertTrue((_ > _)(25, 20))
        self.assertFalse((_ > _)(20, 25))
        self.assertTrue((_ <= _)(6, 7))
        self.assertTrue((_ <= _)(7, 7))
        self.assertFalse((_ <= _)(8, 7))
        self.assertTrue((_ >= _)(8, 7))
        self.assertTrue((_ >= _)(7, 7))
        self.assertFalse((_ >= _)(6, 7))
        self.assertTrue((_ == _)(10, 10))
        self.assertFalse((_ == _)(9, 10))

    def test_comparator_filter(self):
        self.assertEqual([0, 1, 2], list(filter(_ < 5, [0, 1, 2, 10, 11, 12])))

    def test_slicing(self):
        self.assertEqual(0,       (_[0])(list(range(10))))
        self.assertEqual(9,       (_[-1])(list(range(10))))
        self.assertEqual([3, 4, 5], (_[3:])(list(range(6))))
        self.assertEqual([0, 1, 2], (_[:3])(list(range(10))))
        self.assertEqual([1, 2, 3], (_[1:4])(list(range(10))))
        self.assertEqual([0, 2, 4], (_[0:6:2])(list(range(10))))

    def test_slicing_multiple(self):
        self.assertEqual(0, (_[_])(range(10), 0))
        self.assertEqual(8, (_[_ * (-1)])(range(10), 2))

    def test_arity_error(self):
        self.assertRaises(underscore.ArityError, _, 1, 2)
        self.assertRaises(underscore.ArityError, _ + _, 1)
        # can be catched as TypeError
        self.assertRaises(TypeError, _, 1, 2)
        self.assertRaises(TypeError, _ + _, 1)

    def test_more_than_2_operations(self):
        self.assertEqual(12, (_ * 2 + 10)(1))
        self.assertEqual(6,  (_ + _ + _)(1, 2, 3))
        self.assertEqual(10, (_ + _ + _ + _)(1, 2, 3, 4))
        self.assertEqual(7,  (_ + _ * _)(1, 2, 3))

    def test_string_converting(self):
        self.assertEqual("(x1) => x1", str(_))

        self.assertEqual("(x1) => (x1 + 2)",  str(_ + 2))
        self.assertEqual("(x1) => (x1 - 2)",  str(_ - 2))
        self.assertEqual("(x1) => (x1 * 2)",  str(_ * 2))
        self.assertEqual("(x1) => (x1 / 2)",  str(_ / 2))
        self.assertEqual("(x1) => (x1 % 2)",  str(_ % 2))
        self.assertEqual("(x1) => (x1 ** 2)", str(_ ** 2))

        self.assertEqual("(x1) => (x1 & 2)", str(_ & 2))
        self.assertEqual("(x1) => (x1 | 2)", str(_ | 2))
        self.assertEqual("(x1) => (x1 ^ 2)", str(_ ^ 2))

        self.assertEqual("(x1) => (x1 >> 2)", str(_ >> 2))
        self.assertEqual("(x1) => (x1 << 2)", str(_ << 2))

        self.assertEqual("(x1) => (x1 < 2)",  str(_ < 2))
        self.assertEqual("(x1) => (x1 > 2)",  str(_ > 2))
        self.assertEqual("(x1) => (x1 <= 2)", str(_ <= 2))
        self.assertEqual("(x1) => (x1 >= 2)", str(_ >= 2))
        self.assertEqual("(x1) => (x1 == 2)", str(_ == 2))
        self.assertEqual("(x1) => (x1 != 2)", str(_ != 2))

        self.assertEqual("(x1) => ((x1 * 2) + 1)", str((_ * 2 + 1)))

    def test_rigthside_string_converting(self):
        self.assertEqual("(x1) => (2 + x1)",  str(2 + _))
        self.assertEqual("(x1) => (2 - x1)",  str(2 - _))
        self.assertEqual("(x1) => (2 * x1)",  str(2 * _))
        self.assertEqual("(x1) => (2 / x1)",  str(2 / _))
        self.assertEqual("(x1) => (2 % x1)",  str(2 % _))
        self.assertEqual("(x1) => (2 ** x1)", str(2 ** _))

        self.assertEqual("(x1) => (2 & x1)", str(2 & _))
        self.assertEqual("(x1) => (2 | x1)", str(2 | _))
        self.assertEqual("(x1) => (2 ^ x1)", str(2 ^ _))

        self.assertEqual("(x1) => (2 >> x1)", str(2 >> _))
        self.assertEqual("(x1) => (2 << x1)", str(2 << _))

    def test_unary_string_converting(self):
        self.assertEqual("(x1) => (+x1)", str(+_))
        self.assertEqual("(x1) => (-x1)", str(-_))
        self.assertEqual("(x1) => (~x1)", str(~_))

    def test_multiple_string_converting(self):
        self.assertEqual("(x1, x2) => (x1 + x2)", str(_ + _))
        self.assertEqual("(x1, x2) => (x1 * x2)", str(_ * _))
        self.assertEqual("(x1, x2) => (x1 - x2)", str(_ - _))
        self.assertEqual("(x1, x2) => (x1 / x2)", str(_ / _))
        self.assertEqual("(x1, x2) => (x1 % x2)", str(_ % _))
        self.assertEqual("(x1, x2) => (x1 ** x2)", str(_ ** _))

        self.assertEqual("(x1, x2) => (x1 & x2)", str(_ & _))
        self.assertEqual("(x1, x2) => (x1 | x2)", str(_ | _))
        self.assertEqual("(x1, x2) => (x1 ^ x2)", str(_ ^ _))

        self.assertEqual("(x1, x2) => (x1 >> x2)", str(_ >> _))
        self.assertEqual("(x1, x2) => (x1 << x2)", str(_ << _))

        self.assertEqual("(x1, x2) => (x1 > x2)",  str(_ > _))
        self.assertEqual("(x1, x2) => (x1 < x2)",  str(_ < _))
        self.assertEqual("(x1, x2) => (x1 >= x2)", str(_ >= _))
        self.assertEqual("(x1, x2) => (x1 <= x2)", str(_ <= _))
        self.assertEqual("(x1, x2) => (x1 == x2)", str(_ == _))
        self.assertEqual("(x1, x2) => (x1 != x2)", str(_ != _))

        self.assertEqual(
            "(x1, x2) => (((x1 / x2) - 1) * 100)",
            str((_ / _ - 1) * 100)
        )

    def test_reverse_string_converting(self):
        self.assertEqual("(x1, x2, x3) => ((x1 + x2) + x3)", str(_ + _ + _))
        self.assertEqual("(x1, x2, x3) => (x1 + (x2 * x3))", str(_ + _ * _))

        self.assertEqual("(x1) => (1 + (2 * x1))", str((1 + 2 * _)))

    def test_multi_underscore_string_converting(self):
        self.assertEqual("(x1) => (x1 + '_')", str(_ + "_"))
        self.assertEqual(
            "(x1, x2) => getattr((x1 + x2), '__and_now__')",
            str((_ + _).__and_now__)
        )
        self.assertEqual(
            "(x1, x2) => x1['__name__'][x2]",
            str(_['__name__'][_])
        )

    def test_repr(self):
        self.assertEqual(_ / 2, eval(repr(_ / 2)))
        self.assertEqual(_ + _, eval(repr(_ + _)))
        self.assertEqual(_ + _ * _, eval(repr(_ + _ * _)))

    def test_repr_parse_str(self):
        self.assertEqual('=> ' + _, eval(repr('=> ' + _)))
        self.assertEqual(
            reduce(lambda f, n: f.format(n), ('({0} & _)',) * 11).format('_'),
            repr(reduce(_ & _, (_,) * 12)),
        )