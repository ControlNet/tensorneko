from functools import partial, update_wrapper, wraps


def _identity_func(arg):
    return arg


class F:
    """
    A function wrapper for pipe operations and partial function.

    Args:
        f (Callable, optional): A function. Default identity function.
        args, kwargs: The arguments to be passed to the function.

    Examples::

        from tensorneko.util import F

        # compose function
        func = F() << (_ + 10) << (_ + 5)
        print(func(10))  # 25

        # the pipe support the tuple as input
        func = F() >> (filter, _ < 6) >> sum
        print(func(range(10))) # 15

        # partial function
        def add(x, y):
            return x + y
        func = F(add, y=1)
        print(func(10)) # 11

    References:
        GitHub - fnpy/fp.py: Missing features of fp in Python -- active fork of kachayev/fp.py. (2022).
        Retrieved 5 April 2022, from https://github.com/fnpy/fn.py

        GitHub - kachayev/fp.py: Functional programming in Python: implementation of missing features to enjoy
        FP. (2022). Retrieved 5 April 2022, from https://github.com/kachayev/fn.py
    """

    def __init__(self, f=_identity_func, *args, **kwargs):
        if args != () or kwargs != {}:
            f = partial(f, *args, **kwargs)
        self.f = f

    @classmethod
    def __compose(cls, f, g):
        return cls(lambda *args, **kwargs: f(g(*args, **kwargs)))

    def __ensure_callable(self, f):
        return self.__class__(*f) if isinstance(f, tuple) else f

    def __rshift__(self, g):
        """Overload >> operator for F instances"""
        return self.__compose(self.__ensure_callable(g), self.f)

    def __lshift__(self, g):
        """Overload << operator for F instances"""
        return self.__compose(self.f, self.__ensure_callable(g))

    def __call__(self, *args, **kwargs):
        """Overload apply operator"""
        return self.f(*args, **kwargs)


def curry(func):
    """
    A decorator that makes the function curried

    Args:
        func (Callable, optional): A function for currying.

    Examples::

        from tensorneko.util.fp import curried

            @curried
            def sum5(a, b, c, d, e):
                return a + b + c + d + e

            print(sum5(1)(2)(3)(4)(5))  # 15
            print(sum5(1, 2, 3)(4, 5))  # 15
    """

    def _args_len(f):
        from inspect import signature
        args = signature(f).parameters
        return len(args)

    @wraps(func)
    def _curried(*args, **kwargs):
        f = func
        count = 0
        while isinstance(f, partial):
            if f.args:
                count += len(f.args)
            f = f.func

        if count == _args_len(f) - len(args):
            return func(*args, **kwargs)

        para_func = partial(func, *args, **kwargs)
        update_wrapper(para_func, f)
        return curry(para_func)

    return _curried
