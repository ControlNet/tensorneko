from __future__ import annotations

from functools import reduce
from typing import Collection, List, Callable, Union, Iterator, Iterable

from .abstract_seq import AbstractSeq
from ...type import T, R
from ....backend.parallel import ExecutorPool, ParallelType


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

    def map(self, f: Callable[[T], R]) -> Seq[R]:
        return Seq(map(f, self._items))

    def parallel_map(self, f: Callable[[T], R], parallel_type: ParallelType = ParallelType.PROCESS) -> Seq[R]:
        futures = []

        if f.__name__ == "<lambda>":
            raise NotImplementedError("lambda function is not supported yet")

        for item in self._items:
            futures.append(ExecutorPool.submit(f, item, parallel_type=parallel_type))
        return Seq(map(lambda future: future.result(), futures))

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

    def flat_map(self, f: Callable[[T], AbstractSeq[R]]) -> Seq[R]:
        return self.map(f).flatten()

    def take(self, n: int) -> Seq[T]:
        return self[:n]

    def skip(self, n: int) -> Seq[T]:
        return self[n:]

    def to_list(self) -> List[T]:
        return self._items

    def __str__(self):
        return f"Seq({', '.join(map(str, self._items))})"

    __repr__ = __str__
