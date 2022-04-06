from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Union, Callable, List

from ...type import T, R


class AbstractSeq(ABC, Generic[T]):

    @abstractmethod
    def __getitem__(self, item: Union[int, slice]) -> Union[T, AbstractSeq[T]]:
        ...

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def map(self, f: Callable[[T], R]) -> AbstractSeq[R]:
        ...

    @abstractmethod
    def filter(self, f: Callable[[T], bool]) -> AbstractSeq[T]:
        ...

    @abstractmethod
    def reduce(self, f: Callable[[T, T], T]) -> T:
        ...

    @abstractmethod
    def flatten(self) -> AbstractSeq[T]:
        ...

    @abstractmethod
    def flat_map(self, f: Callable[[T], AbstractSeq[R]]) -> AbstractSeq[R]:
        ...

    @abstractmethod
    def skip(self, n: int) -> AbstractSeq[T]:
        ...

    @abstractmethod
    def take(self, n: int) -> AbstractSeq[T]:
        ...

    @abstractmethod
    def to_list(self) -> List[T]:
        ...
