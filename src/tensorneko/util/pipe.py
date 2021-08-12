from __future__ import annotations

from typing import Callable, Union, Tuple

from fn import F

from ..util import dict_add


class Args(F):

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
            return Args(*(self.args + g.args), **(dict_add(self.kwargs, g.kwargs)))
        elif isinstance(g, tuple):
            return self.__ensure_callable(g)
        elif any(map(lambda getter: g is getter, (Args.get, Args.get_args, Args.get_value, Args.get_kwargs))):
            return g(self)
        else:
            return Args(g(*self.args, **self.kwargs))

    def __call__(self, *args, **kwargs):
        """Overload apply operator"""
        raise TypeError("The 'Args' is not callable")

    def __repr__(self):
        kwargs_str = ""
        for k, v in self.kwargs.items():
            kwargs_str = f"{kwargs_str}, {k}={v}"

        return f"({', '.join(map(str, self.args))}{kwargs_str})"

    def get(self) -> any:
        return self.args[0]

    def get_args(self) -> tuple:
        return self.args

    def get_kwargs(self) -> dict:
        return self.kwargs

    def get_value(self, key: str) -> dict:
        return self.kwargs[key]