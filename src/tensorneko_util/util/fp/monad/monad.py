from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Callable, TypeVar

from ...type import T_CO as T, R


class Monad(Generic[T], ABC):

    # method `pure` is equivalent to the constructor

    @abstractmethod
    def flat_map(self, f: Callable[[T], Monad[R]]) -> Monad[R]:
        ...

    @abstractmethod
    def map(self, f: Callable[[T], R]) -> Monad[R]:
        ...

    @abstractmethod
    def flatten(self: Monad[Monad[T]]) -> Monad[T]:
        ...
