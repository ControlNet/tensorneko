from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Callable, List, Optional, Iterable

from ...type import T, R
from ....backend._tqdm import import_tqdm_auto
from ....backend.parallel import ParallelType


class AbstractSeq(Iterable[T], ABC):

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
    def for_each(self, f: Callable[[T], None], progress_bar: bool = False, parallel_type: Optional[ParallelType] = None
    ) -> None:
        ...

    @abstractmethod
    def with_for_each(self, f: Callable[[T], None], progress_bar: bool = False,
        parallel_type: Optional[ParallelType] = None
    ) -> AbstractSeq[T]:
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

    @property
    @abstractmethod
    def head(self) -> T:
        ...

    @property
    @abstractmethod
    def tail(self) -> AbstractSeq[T]:
        ...

    @abstractmethod
    def repeat(self, n: int) -> AbstractSeq[T]:
        ...

    @abstractmethod
    def to_list(self) -> List[T]:
        ...

    @property
    def _tqdm(self):
        return import_tqdm_auto().tqdm
