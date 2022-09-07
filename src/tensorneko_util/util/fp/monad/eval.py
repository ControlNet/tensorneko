from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .monad import Monad
from ...type import T_CO as T, R


class Eval(Monad[T], ABC):
    """
    ``Eval`` is a monad that encapsulates the evaluation of a value.

    Example:

        Use `Eval.now` to create an ``Eval`` object that contains the value which is evaluated immediately.

        .. code-block:: python

            from tensorneko.util import Eval

            @Eval.now
            def x():
                print("Eval x")
                return 1  # Eval x

            def f1(value: int) -> int:
                print("Eval f1")
                return value + 1

            y = x.map(f1)  # Eval f1
            print(y.value)  # 2
            print(y.value)  # 2
            print(y.value)  # 2

        Use `Eval.later` to create an ``Eval`` object that contains the value which is evaluated lazily.

        .. code-block:: python

            from tensorneko.util import Eval

            @Eval.later
            def x():
                print("Eval x")
                return 1

            def f1(value: int) -> int:
                print("Eval f1")
                return value + 1

            y = x.map(f1)  # Eval x, Eval f1
            print(y.value)  # 2
            print(y.value)  # 2
            print(y.value)  # 2

        Use `Eval.always` to create an ``Eval`` object that contains the value which will be evaluated when `Eval.value`
            is called.

        .. code-block:: python

            from tensorneko.util import Eval

            @Eval.always
            def x():
                print("Eval x")
                return 1

            def f1(value: int) -> int:
                print("Eval f1")
                return value + 1

            y = x.map(f1)
            print(y.value)  # Eval x, Eval f1, 2
            print(y.value)  # Eval x, Eval f1, 2
            print(y.value)  # Eval x, Eval f1, 2
    """

    def __init__(self: T_E, eval_func: Callable[[], T]):
        self._getter = eval_func
        self._value: Optional[T] = None

    @property
    @abstractmethod
    def value(self) -> T:
        ...

    @classmethod
    def pure(cls, value: T) -> Eval[T]:
        return cls(lambda: value)

    def map(self, f: Callable[[T], R]) -> Eval[R]:
        return self.__class__(lambda: f(self.value))

    def flat_map(self, f: Callable[[T], Eval[R]]) -> Eval[R]:
        return self.__class__(lambda: f(self.value).value)

    def flatten(self: Eval[Eval[T]]) -> Eval[T]:
        return self.value


class Always(Eval[T]):
    """Call by name"""

    @property
    def value(self) -> T:
        return self._getter()


class Later(Eval[T]):
    """Call by need"""

    @property
    def value(self) -> T:
        if self._value is None:
            self._value = self._getter()
        return self._value


class Now(Eval[T]):
    """Call by value"""

    def __init__(self, eval_func: Callable[[], T]):
        super().__init__(eval_func)
        self._value = self._getter()

    @property
    def value(self) -> T:
        return self._value


def _always(eval_func: Callable[[], T]) -> Always[T]:
    return Always(eval_func)


def _later(eval_func: Callable[[], T]) -> Later[T]:
    return Later(eval_func)


def _now(eval_func: Callable[[], T]) -> Now[T]:
    return Now(eval_func)


# force these factory function does not appear in these subclasses
Eval.always = _always
Eval.later = _later
Eval.now = _now
