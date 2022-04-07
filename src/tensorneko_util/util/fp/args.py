from __future__ import annotations

from typing import Union, Callable, Tuple, Any

from .func import F


class Args(F):
    """
    A variable wrapper for pipe operations.

    Args:
        args, kwargs: The arguments to be passed to the pipe.

    Examples::

        from tensorneko.util import __, _
        result = __(20) >> (_ + 1) >> (_ * 2) >> __.get
        print(result)
        # 42

    References:
        GitHub - fnpy/fp.py: Missing features of fp in Python -- active fork of kachayev/fp.py. (2022).
        Retrieved 5 April 2022, from https://github.com/fnpy/fn.py

        GitHub - kachayev/fp.py: Functional programming in Python: implementation of missing features to enjoy
        FP. (2022). Retrieved 5 April 2022, from https://github.com/kachayev/fn.py

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def __ensure_callable(self, inputs):
        return Args(*(self.args + inputs), **self.kwargs) if isinstance(inputs, tuple) else inputs

    def __lshift__(self, g):
        """Overload << operator for Input and F instances"""
        raise ValueError("The Args instance cannot be the end of a pipe.")

    def __rshift__(self, g: Union[Callable, Tuple, Args, F]) -> Union[Args, any]:
        """Overload >> operator for F instances"""
        if type(g) is Args:
            self.kwargs.update(g.kwargs)
            return Args(*(self.args + g.args), **self.kwargs)
        elif isinstance(g, tuple):
            return self.__ensure_callable(g)
        elif any(map(lambda getter: g is getter, (Args.get, Args.get_args, Args.get_value, Args.get_kwargs))):
            return g(self)
        else:
            return Args(g(*self.args, **self.kwargs))

    def __call__(self, *args, **kwargs):
        """Overload apply operator"""
        raise TypeError("The 'Args' is not callable")

    def __str__(self):
        kwargs_str = ""
        for k, v in self.kwargs.items():
            kwargs_str = f"{kwargs_str}, {k}={v}"

        return f"({', '.join(map(str, self.args))}{kwargs_str})"

    __repr__ = __str__

    def get(self) -> Any:
        return self.args[0]

    def get_args(self) -> tuple:
        return self.args

    def get_kwargs(self) -> dict:
        return self.kwargs

    def get_value(self, key: str) -> dict:
        return self.kwargs[key]
