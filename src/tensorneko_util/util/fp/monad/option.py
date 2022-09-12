from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial, wraps
from operator import eq, is_not
from typing import Callable, Union, Optional, Any, Tuple, List

from .monad import Monad
from ...singleton import Singleton
from ...type import T_CO as T, R, T1, T2, T3


class Option(Monad[T], ABC):
    """
    ``Option`` represents optional values, each instance of ``Option``
    can be either instance of ``Some`` or ``Empty``.
    """
    empty: bool

    def __new__(cls, value: Union[T, Option[T]], checker: Callable[[T], bool] = partial(is_not, None)) -> Option[T]:
        if isinstance(value, Option):
            # Option(Some) -> Some
            # Option(Empty) -> Empty
            return value

        return Some.__new__(Some, value) if checker(value) else Empty

    def is_empty(self) -> bool:
        return self.empty

    def is_defined(self) -> bool:
        return not self.is_empty()

    @abstractmethod
    def flatten(self: Option[Option[T]]) -> Option[T]:
        ...

    @abstractmethod
    def map(self, f: Callable[[T], R]) -> Option[R]:
        ...

    @abstractmethod
    def flat_map(self, f: Callable[[T], Option[R]]) -> Option[R]:
        ...

    @abstractmethod
    def fold(self, f: Callable[[T], R], if_empty: R) -> R:
        ...

    @abstractmethod
    def for_each(self, f: Callable[[T], None]) -> None:
        ...

    @abstractmethod
    def filter(self, f: Callable[[T], bool]) -> Option[T]:
        ...

    @abstractmethod
    def filter_not(self, f: Callable[[T], bool]) -> Option[T]:
        ...

    @abstractmethod
    def exists(self, f: Callable[[T], bool]) -> bool:
        ...

    @abstractmethod
    def for_all(self, f: Callable[[T], bool]) -> bool:
        ...

    @abstractmethod
    def contains(self, value: T) -> bool:
        ...

    @abstractmethod
    def zip(self, other: Option[R]) -> Option[Tuple[T, R]]:
        ...

    @abstractmethod
    def unzip(self: Option[Tuple[T1, T2]]) -> Tuple[Option[T1], Option[T2]]:
        ...

    @abstractmethod
    def unzip3(self: Option[Tuple[T1, T2, T3]]) -> Tuple[Option[T1], Option[T2], Option[T3]]:
        ...

    @abstractmethod
    def get(self) -> Optional[T]:
        ...

    @abstractmethod
    def get_or_else(self, default: T) -> T:
        ...

    @abstractmethod
    def get_or_call(self, f: Callable[[...], T], *args, **kwargs) -> T:
        ...

    @abstractmethod
    def or_else(self, default: T) -> Option[T]:
        ...

    @abstractmethod
    def or_call(self, f: Callable[[...], T], *args, **kwargs) -> Option[T]:
        ...

    @abstractmethod
    def to_list(self) -> List[T]:
        ...


class Some(Option[T]):

    empty = False

    def __init__(self, value: T, checker=None):
        # Option(Some) -> Some
        self.x: T = value.get() if isinstance(value, Some) else value

    def __new__(cls, value: T):
        return object.__new__(cls)

    def flatten(self: Some[Option[T]]) -> Option[T]:
        return self.x

    def map(self, f: Callable[[T], R]) -> Option[R]:
        return Option(f(self.x))

    def flat_map(self, f: Callable[[T], Option[R]]) -> Option[R]:
        return f(self.x)

    def fold(self, f: Callable[[T], R], if_empty: R) -> R:
        return f(self.x)

    def for_each(self, f: Callable[[T], None]) -> None:
        f(self.x)

    def filter(self, f: Callable[[T], bool]) -> Some[T]:
        return self if f(self.x) else Empty

    def filter_not(self, f: Callable[[T], bool]) -> Option[T]:
        return self if not f(self.x) else Empty

    def exists(self, f: Callable[[T], bool]) -> bool:
        return f(self.x)

    def for_all(self, f: Callable[[T], bool]) -> bool:
        return f(self.x)

    def contains(self, value: T) -> bool:
        return eq(self.x, value)

    def zip(self, other: Option[R]) -> Option[Tuple[T, R]]:
        if other.is_empty():
            return Empty
        return Some((self.x, other.get()))

    def unzip(self: Some[Tuple[T1, T2]]) -> Tuple[Option[T1], Option[T2]]:
        if len(self.x) != 2 or not isinstance(self.x, tuple):
            raise ValueError("Pair tuple expected")
        a, b = self.x
        return Some(a), Some(b)

    def unzip3(self: Some[Tuple[T1, T2, T3]]) -> Tuple[Option[T1], Option[T2], Option[T3]]:
        a, b, c = self.x
        return Some(a), Some(b), Some(c)

    def get(self) -> T:
        return self.x

    def get_or_else(self, default: T) -> T:
        return self.x

    def get_or_call(self, f: Callable[[...], T], *args, **kwargs) -> T:
        return self.x

    def or_else(self, default: T) -> Some[T]:
        return self

    def or_call(self, f: Callable[[...], T], *args, **kwargs) -> Some[T]:
        return self

    def to_list(self) -> List[T]:
        return [self.x]

    def __str__(self) -> str:
        return "Some(%s)" % self.x

    __repr__ = __str__

    def __eq__(self, other: Option[Any]) -> bool:
        if not isinstance(other, Some):
            return False
        return eq(self.x, other.x)


@Singleton
class Empty(Option[None]):

    empty = True

    def __new__(cls) -> Empty:
        return object.__new__(cls)

    def flatten(self: Empty) -> Empty:
        return Empty

    def map(self, f: Callable[[T], R]) -> Empty:
        return Empty

    def flat_map(self, f: Callable[[T], Option[R]]) -> Option[R]:
        return Empty

    def fold(self, f: Callable[[T], R], if_empty: R) -> R:
        return if_empty

    def for_each(self, f: Callable[[T], None]) -> None:
        return None

    def filter(self, f: Callable[[T], bool]) -> Empty:
        return Empty

    def filter_not(self, f: Callable[[T], bool]) -> Option[T]:
        return Empty

    def exists(self, f: Callable[[T], bool]) -> bool:
        return False

    def for_all(self, f: Callable[[T], bool]) -> bool:
        return True

    def contains(self, value: T) -> bool:
        return False

    def zip(self, other: Option[R]) -> Empty:
        return Empty

    def unzip(self: Empty) -> Tuple[Empty, Empty]:
        return Empty, Empty

    def unzip3(self: Empty) -> Tuple[Empty, Empty, Empty]:
        return Empty, Empty, Empty

    def get_or_else(self, default: T) -> T:
        return default

    def get(self) -> None:
        return None

    def get_or_call(self, f: Callable[[...], T], *args, **kwargs) -> T:
        return f(*args, **kwargs)

    def or_else(self, default: T) -> Option[R]:
        return Option(default)

    def or_call(self, f: Callable[[...], T], *args, **kwargs) -> Option[T]:
        return Option(f(*args, **kwargs))

    def to_list(self) -> List:
        return []

    def __str__(self) -> str:
        return "Empty"

    __repr__ = __str__

    def __eq__(self, other: Option[Any]) -> bool:
        return other is Empty


class ReturnOptionDecorator:
    """
    Decorator for modifying functions to return Option.
    It will return ``Empty`` if the return value doesn't meet checker.
    """

    def __init__(self, checker: Callable[[T], bool] = partial(is_not, None)):
        self._checker = checker

    def __call__(self, func: Callable[[...], T]) -> Callable[[...], Option[T]]:
        """
        Decorator for modifying functions to return Option.
        In default, it uses ``lambda x: x is not None`` as checker, when you use ``@return_option``.
        If you want to use your own checker, you can use ``@return_option.with_checker(checker)``.

        Args:
            func(``(...) -> T``): function to be modified to return Option[T].

        Returns:
            func(``(...) -> Option[T]``): modified function.

        Examples::

            from tensorneko.util import return_option

            @return_option
            def f1(x: int) -> int:
                return x + 1

            print(f1(1))  # Some(2)

            @return_option.with_checker(lambda x: x > 0)
            def f2(x: int) -> int:
                return x - 10

            print(f2(1))  # Empty

        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Option[T]:
            res = func(*args, **kwargs)
            return Option(res, checker=self._checker)

        return wrapper

    @classmethod
    def with_checker(cls, checker: Callable[[T], bool]) -> Callable[[Callable[[...], T]], Callable[[...], Option[T]]]:
        return cls(checker)


return_option = ReturnOptionDecorator()
