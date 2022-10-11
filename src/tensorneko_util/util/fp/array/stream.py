from __future__ import annotations

from functools import reduce
from itertools import chain
from sys import maxsize
from typing import List, Callable, Optional, Union, Iterable, Generator

from .abstract_seq import AbstractSeq
from ...type import T, R
from ....backend.parallel import ParallelType, ExecutorPool


class Stream(AbstractSeq[T]):

    def __init__(self, iterable: Iterable[T] = iter(())):
        self._iter = iter(iterable)
        self._cache: List[T] = []
        self._finished = False
        if isinstance(iterable, Stream):
            self._cache = iterable._cache

    @staticmethod
    def of(*items: T) -> Stream[T]:
        return Stream(items)

    @staticmethod
    def from_stream(stream: Stream[T]) -> Stream[T]:
        stream = Stream(stream._iter)
        stream._cache = stream._cache
        return stream

    def __getitem__(self, item: Union[int, slice]) -> Union[T, AbstractSeq[T]]:
        if isinstance(item, int):
            if item < 0:
                raise ValueError("Negative slice is not allowed in Stream")
            for _ in self._iter_to(item):
                pass

        elif isinstance(item, slice):
            low, high, step = item.indices(maxsize)
            if step == 0:
                raise ValueError("Step must not be 0")
            return self.__class__() << map(self.__getitem__, range(low, high, step or 1))
        else:
            raise TypeError("Invalid argument type")

        return self._cache.__getitem__(item)

    def __lshift__(self, right_iter: Iterable[T]) -> Stream[T]:
        self._iter = chain(self._iter, right_iter)
        return self

    def __iter__(self) -> Generator[T, None, None]:
        return self._iter_all()

    def _iter_to(self, index: int, start: int = 0) -> Generator[T, None, None]:
        return self._iter_cond(lambda i: i <= index, start)

    def _iter_all(self) -> Generator[T, None, None]:
        return self._iter_cond(lambda i: True)

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        i = start
        while cond(i):
            if self._cache_size > i:
                yield self._cache[i]
            elif self._cache_size == i:
                try:
                    item = next(self._iter)
                except StopIteration:
                    self._finished = True
                    break
                self._cache.append(item)
                yield item
            else:
                for item in self._iter_to(i, start=self._cache_size):
                    yield item
            i += 1

    @property
    def _cache_size(self) -> int:
        return len(self._cache)

    def map(self, f: Callable[[T], R]) -> Stream[R]:
        return MapStream(self, f)

    def for_each(self, f: Callable[[T], None], progress_bar: bool = False, parallel_type: Optional[ParallelType] = None,
        **tqdm_args
    ) -> None:
        if parallel_type is None:
            items = self if not progress_bar else self._tqdm(self, **tqdm_args)
            for item in items:
                f(item)
        else:
            futures = []

            if f.__name__ == "<lambda>":
                raise NotImplementedError("lambda function is not supported yet")

            for item in self:
                futures.append(ExecutorPool.submit(f, item, parallel_type=parallel_type))

            futures = self._tqdm(futures, **tqdm_args) if progress_bar else futures
            for future in futures:
                future.result()

    def with_for_each(self, f: Callable[[T], None], progress_bar: bool = False,
        parallel_type: Optional[ParallelType] = None, **tqdm_args
    ) -> Stream[T]:
        self.for_each(f, progress_bar, parallel_type, **tqdm_args)
        return self

    def filter(self, f: Callable[[T], bool]) -> Stream[T]:
        return FilterStream(self, f)

    def reduce(self, f: Callable[[T, T], T]) -> T:
        return reduce(f, self)

    def flatten(self) -> Stream[T]:
        return FlattenStream(self)

    def flat_map(self, f: Callable[[T], Stream[R]]) -> Stream[R]:
        return self.map(f).flatten()

    def skip(self, n: int) -> Stream[T]:
        return SkipStream(self, n)

    def take(self, n: int) -> Stream[T]:
        return TakeStream(self, n)

    def repeat(self, n: int) -> Stream[T]:
        return RepeatStream(self, n)

    @property
    def head(self) -> T:
        return self[0]

    @property
    def tail(self) -> Stream[T]:
        return self.skip(1)

    def to_list(self) -> List[T]:
        return [item for item in self]

    def __str__(self):
        return f"Stream({', '.join(map(str, self._cache + (['...'] if not self._finished else [])))})"


class MapStream(Stream[T]):

    def __init__(self, iterable: Iterable[T], f: Callable[[T], R]):
        super().__init__(iterable)
        self._f = f
        self._cache = []

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        i = start
        while cond(i):
            if self._cache_size > i:
                yield self._cache[i]
            elif self._cache_size == i:
                try:
                    item = self._f(next(self._iter))
                except StopIteration:
                    break
                self._cache.append(item)
                yield item
            else:
                for item in self._iter_to(i, start=self._cache_size):
                    yield item
            i += 1


class FilterStream(Stream[T]):

    def __init__(self, iterable: Iterable[T], f: Callable[[T], bool]):
        super().__init__(iterable)
        self._f = f
        self._cache = []

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        i = start
        while cond(i):
            if self._cache_size > i:
                yield self._cache[i]
                i += 1
            elif self._cache_size == i:
                try:
                    item = next(self._iter)
                except StopIteration:
                    break
                if self._f(item):
                    self._cache.append(item)
                    yield item
                    i += 1
            else:
                for item in self._iter_to(i, start=self._cache_size):
                    yield item
                    i += 1


class FlattenStream(Stream[T]):

    def __init__(self, iterable: Iterable[T]):
        super().__init__(iterable)
        self._cache = []

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        i = start
        while cond(i):
            if self._cache_size > i:
                yield self._cache[i]
                i += 1
            elif self._cache_size == i:
                try:
                    item = next(self._iter)
                except StopIteration:
                    break
                if isinstance(item, Stream):
                    for each in item:
                        self._cache.append(each)
                        yield each
                        i += 1
                else:
                    self._cache.append(item)
                    yield item
                    i += 1
            else:
                for item in self._iter_to(i, start=self._cache_size):
                    yield item
                    i += 1


class TakeStream(Stream[T]):

    def __init__(self, iterable: Iterable[T], n: int):
        super().__init__(iterable)
        self._n = n
        self._cache = []

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        def _cond(i: int) -> bool:
            return cond(i) and i < self._n

        return super()._iter_cond(_cond, start=start)


class SkipStream(Stream[T]):

    def __init__(self, iterable: Iterable[T], n: int):
        super().__init__(iterable)
        self._n = n
        self._cache = []
        self._skip_applied = False

    def _apply_skip(self) -> None:
        for _ in range(self._n):
            next(self._iter)
        self._skip_applied = True

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        if not self._skip_applied:
            self._apply_skip()
        return super()._iter_cond(cond, start=start)


class RepeatStream(Stream[T]):

    def __init__(self, iterable: Iterable[T], n: int):
        super().__init__(iterable)
        self._cache = []
        self._n = n
        self._iter_once = False

    def _iter_cond(self, cond: Callable[[int], bool], start: int = 0) -> Generator[T, None, None]:
        i = start
        while cond(i):
            if not self._iter_once:
                if self._cache_size > i:
                    yield self._cache[i]
                elif self._cache_size == i:
                    try:
                        item = next(self._iter)
                    except StopIteration:
                        self._iter_once = True
                        continue
                    self._cache.append(item)
                    yield item
                else:
                    for item in self._iter_to(i, start=self._cache_size):
                        yield item

            else:
                if i < len(self._cache) * self._n:
                    yield self._cache[i % self._cache_size]
                else:
                    break
            i += 1
