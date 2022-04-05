# edit from https://github.com/fnpy/fn.py/blob/master/fn/underscore.py
# refactor later

import operator
import random
import re
import string
from itertools import count, repeat

from .func import F, _identity_func as identity


def _flip(f):
    """Return function that will apply arguments in reverse order"""

    # Original function is saved in special attribute
    # in order to optimize operation of "duble flipping",
    # so flip(flip(A)) is A
    # Do not use this approach for underscore callable,
    # see https://github.com/kachayev/fn.py/issues/23
    flipper = getattr(f, "__flipback__", None)
    if flipper is not None:
        return flipper

    def _flipper(a, b):
        return f(b, a)

    setattr(_flipper, "__flipback__", f)
    return _flipper


def _random_name():
    return "".join(random.choice(string.ascii_letters) for _ in range(14))


def _fmap(f, pattern):
    def applier(self, other):
        fmt = "(%s)" % pattern.replace("self", self._format)

        if isinstance(other, self.__class__):
            return self.__class__(
                (f, self, other),
                fmt.replace("other", other._format),
                dict(
                    list(self._format_args.items()) +
                    list(other._format_args.items())
                ),
                self._arity + other._arity
            )
        else:
            call = F(_flip(f), other) << F(self)
            name = _random_name()
            return self.__class__(
                call,
                fmt.replace("other", "%%(%s)r" % name),
                dict(list(self._format_args.items()) + [(name, other)]),
                self._arity)
    return applier


class ArityError(TypeError):
    def __str__(self):
        return "{0!r} expected {1} arguments, got {2}".format(*self.args)


def _unary_fmap(f, format):
    def applier(self):
        fmt = "(%s)" % format.replace("self", self._format)
        return self.__class__(
            F(self) << f, fmt, self._format_args, self._arity
        )
    return applier


class _Callable:

    __slots__ = "_callback", "_format", "_format_args", "_arity"
    # Do not use "flipback" approach for underscore callable,
    # see https://github.com/kachayev/fn.py/issues/23
    __flipback__ = None

    def __init__(self, callback=identity, format="_", format_args=None,
                 arity=1):
        self._callback = callback
        self._format = format
        self._format_args = format_args or {}
        self._arity = arity

    def call(self, name, *args, **kwargs):
        """Call method from _ object by given name and arguments"""
        return self.__class__(
            F(lambda f: f(*args, **kwargs)) << operator.attrgetter(name) << F(self)
        )

    def __getattr__(self, name):
        if name == '__wrapped__':  # Guard for recursive call by doctest
            raise AttributeError
        attr_name = _random_name()
        return self.__class__(
            F(operator.attrgetter(name)) << F(self),
            "getattr(%s, %%(%s)r)" % (self._format, attr_name),
            dict(
                list(self._format_args.items()) + [(attr_name, name)]
            ),
            self._arity
        )

    def __getitem__(self, k):
        if isinstance(k, self.__class__):
            return self.__class__(
                (operator.getitem, self, k),
                "%s[%s]" % (self._format, k._format),
                dict(
                    list(self._format_args.items()) +
                    list(k._format_args.items())
                ),
                self._arity + k._arity
            )
        item_name = _random_name()
        return self.__class__(
            F(operator.itemgetter(k)) << F(self),
            "%s[%%(%s)r]" % (self._format, item_name),
            dict(list(self._format_args.items()) + [(item_name, k)]),
            self._arity
        )

    def __str__(self):
        """Build readable representation for function
        (_ < 7): (x1) => (x1 < 7)
        (_ + _*10): (x1, x2) => (x1 + (x2*10))
        """
        # args iterator with produce infinite sequence
        # args -> (x1, x2, x3, ...)
        args = map("".join, zip(repeat("x"), map(str, count(1))))
        l, r = [], self._format
        # replace all "_" signs from left to right side
        while r.count("_"):
            n = next(args)
            r = r.replace("_", n, 1)
            l.append(n)

        r = r % self._format_args
        return "({left}) => {right}".format(left=", ".join(l), right=r)

    def __repr__(self):
        """
        Return original function notation to ensure that eval(repr(f)) == f
        """
        return re.sub(r"x\d+", "_", str(self).split("=>", 1)[1].strip())

    def __call__(self, *args):
        if len(args) != self._arity:
            raise ArityError(self, self._arity, len(args))

        if not isinstance(self._callback, tuple):
            return self._callback(*args)

        f, left, right = self._callback
        return f(left(*args[:left._arity]), right(*args[left._arity:]))

    __add__ = _fmap(operator.add, "self + other")
    __mul__ = _fmap(operator.mul, "self * other")
    __sub__ = _fmap(operator.sub, "self - other")
    __mod__ = _fmap(operator.mod, "self %% other")
    __pow__ = _fmap(operator.pow, "self ** other")

    __and__ = _fmap(operator.and_, "self & other")
    __or__ = _fmap(operator.or_, "self | other")
    __xor__ = _fmap(operator.xor, "self ^ other")

    __div__ = _fmap(operator.truediv, "self / other")
    __divmod__ = _fmap(divmod, "self / other")
    __floordiv__ = _fmap(operator.floordiv, "self / other")
    __truediv__ = _fmap(operator.truediv, "self / other")

    __lshift__ = _fmap(operator.lshift, "self << other")
    __rshift__ = _fmap(operator.rshift, "self >> other")

    __lt__ = _fmap(operator.lt, "self < other")
    __le__ = _fmap(operator.le, "self <= other")
    __gt__ = _fmap(operator.gt, "self > other")
    __ge__ = _fmap(operator.ge, "self >= other")
    __eq__ = _fmap(operator.eq, "self == other")
    __ne__ = _fmap(operator.ne, "self != other")

    __neg__ = _unary_fmap(operator.neg, "-self")
    __pos__ = _unary_fmap(operator.pos, "+self")
    __invert__ = _unary_fmap(operator.invert, "~self")

    __radd__ = _fmap(_flip(operator.add), "other + self")
    __rmul__ = _fmap(_flip(operator.mul), "other * self")
    __rsub__ = _fmap(_flip(operator.sub), "other - self")
    __rmod__ = _fmap(_flip(operator.mod), "other %% self")
    __rpow__ = _fmap(_flip(operator.pow), "other ** self")
    __rdiv__ = _fmap(_flip(operator.truediv), "other / self")
    __rdivmod__ = _fmap(_flip(divmod), "other / self")
    __rtruediv__ = _fmap(_flip(operator.truediv), "other / self")
    __rfloordiv__ = _fmap(_flip(operator.floordiv), "other / self")

    __rlshift__ = _fmap(_flip(operator.lshift), "other << self")
    __rrshift__ = _fmap(_flip(operator.rshift), "other >> self")

    __rand__ = _fmap(_flip(operator.and_), "other & self")
    __ror__ = _fmap(_flip(operator.or_), "other | self")
    __rxor__ = _fmap(_flip(operator.xor), "other ^ self")


shortcut = _Callable()
