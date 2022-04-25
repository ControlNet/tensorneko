from __future__ import annotations

from functools import reduce
from typing import Collection, List, Callable, Union, Iterator, Iterable, Optional, Any
from tqdm.auto import tqdm

from .abstract_seq import AbstractSeq
from ...type import T, R
from ....backend.parallel import ExecutorPool, ParallelType


def _identity(x: T) -> T:
    return x


class Seq(AbstractSeq, Collection[T]):

    def __init__(self, items: Iterable[T]):
        if isinstance(items, Seq):
            self._items = items.to_list()
        else:
            self._items = list(items)

    @staticmethod
    def of(*item: T) -> Seq[T]:
        return Seq(item)

    @staticmethod
    def from_seq(seq: Seq[T]) -> Seq[T]:
        return Seq(seq)

    def __lshift__(self, right_iter: Iterable[T]) -> Seq[T]:
        return Seq(self._items + list(right_iter))

    def __getitem__(self, item: Union[int, slice]) -> Union[T, AbstractSeq[T]]:
        if isinstance(item, int):
            return self._items[item]
        elif isinstance(item, slice):
            return Seq(self._items[item])
        else:
            raise TypeError(f"{type(item)} is not supported")

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, __x: object) -> bool:
        return self._items.__contains__(__x)

    def map(self, f: Callable[[T], R], progress_bar: bool = False, parallel_type: Optional[ParallelType] = None
    ) -> Seq[R]:
        if parallel_type is None:
            items = self._items if not progress_bar else tqdm(self._items)
            return Seq(map(f, items))
        else:
            futures = []

            if f.__name__ == "<lambda>":
                raise NotImplementedError("lambda function is not supported yet")

            for item in self._items:
                futures.append(ExecutorPool.submit(f, item, parallel_type=parallel_type))

            futures = tqdm(futures) if progress_bar else futures
            return Seq(map(lambda future: future.result(), futures))

    def for_each(self, f: Callable[[T], None], progress_bar: bool = False, parallel_type: Optional[ParallelType] = None
    ) -> None:
        if parallel_type is None:
            items = self._items if not progress_bar else tqdm(self._items)
            for item in items:
                f(item)
        else:
            self.map(f, progress_bar, parallel_type)

    def with_for_each(self, f: Callable[[T], None], progress_bar: bool = False,
        parallel_type: Optional[ParallelType] = None
    ) -> Seq[T]:
        self.for_each(f, progress_bar, parallel_type)
        return self

    def filter(self, f: Callable[[T], bool]) -> Seq[T]:
        return Seq(filter(f, self._items))

    def reduce(self, f: Callable[[T, T], T]) -> T:
        return reduce(f, self._items)

    def flatten(self) -> Seq[T]:
        new_items = []
        for item in self._items:
            if isinstance(item, Seq):
                new_items.extend(item.to_list())
            else:
                new_items.append(item)
        return Seq(new_items)

    def flat_map(self, f: Callable[[T], AbstractSeq[R]], progress_bar: bool = False) -> Seq[R]:
        return self.map(f, progress_bar).flatten()

    def sort(self, key: Callable[[T], Any] = _identity, reverse: bool = False) -> Seq[T]:
        return Seq(sorted(self._items, key=key, reverse=reverse))

    def take(self, n: int) -> Seq[T]:
        return self[:n]

    def skip(self, n: int) -> Seq[T]:
        return self[n:]

    def to_list(self) -> List[T]:
        return self._items

    def __str__(self):
        return f"Seq({', '.join(map(str, self._items))})"

    __repr__ = __str__
