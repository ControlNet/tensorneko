from __future__ import annotations

from enum import Enum
from functools import reduce
from itertools import chain, islice
from sys import maxsize
from typing import Iterable, Iterator, List, Union, Callable, Optional

from .abstract_seq import AbstractSeq
from ...type import T, R


class Stream(AbstractSeq, Iterable[T]):

    def __init__(self, iterable: Iterable[T] = iter(()), action_pipe: List[_StreamAction] = None):
        if isinstance(iterable, Stream):
            self._iter = iterable._iter
            self._action_pipe = action_pipe or iterable._action_pipe
            self._cache = iterable._cache
            self._current_index = iterable._current_index
        else:
            self._iter = iter(iterable)
            self._action_pipe = action_pipe or []
            self._cache = []
            self._current_index = -1

    @staticmethod
    def of(*items: T) -> Stream[T]:
        return Stream(items)

    @staticmethod
    def from_stream(stream: Stream[T]) -> Stream[T]:
        return Stream(stream._iter)

    def __lshift__(self, right_iter: Iterable[T]) -> Stream[T]:
        self._iter = chain(self._iter, right_iter)
        return self

    def __iter__(self) -> _Iterator:
        return self._Iterator(self)

    class _Iterator(Iterator[T]):

        def __init__(self, stream: Stream[T]):
            self._stream = stream
            self._iter_index = -1

        def __next__(self) -> T:
            self._iter_index += 1
            result = self._stream._try_iter_to(self._iter_index)
            if result in (self._stream._IterStatus.SUCCESS_ITER, self._stream._IterStatus.NO_NEED_ITER):
                return self._stream._cache[self._iter_index]
            else:
                raise StopIteration

    class _IterStatus(Enum):
        FAIL = 0
        NO_NEED_ITER = 1
        SUCCESS_ITER = 2

    def _try_iter_to(self, index: int) -> _IterStatus:
        if index <= self._current_index:
            return self._IterStatus.NO_NEED_ITER

        while index > self._current_index:
            try:
                item = next(self._iter)
            except StopIteration:
                return self._IterStatus.FAIL

            filtered = True
            flattened = False
            for i, (f, method) in enumerate(self._action_pipe):
                if method == Stream.map:
                    item = f(item)
                elif method == Stream.filter:
                    filtered = f(item)
                    if not filtered:
                        break
                elif method == Stream.flatten:
                    if isinstance(item, Stream):
                        item = Stream(item, action_pipe=self._action_pipe[i + 1:]).to_list()
                        flattened = True
                        break  # the remain actions are already applied in the sub stream to_list

            if filtered:
                if flattened:
                    self._cache.extend(item)
                    self._current_index += len(item)
                else:
                    self._cache.append(item)
                    self._current_index += 1

        return self._IterStatus.SUCCESS_ITER

    def __getitem__(self, item: Union[int, slice]) -> Union[T, Stream[T]]:
        if isinstance(item, int):
            if item < 0:
                raise ValueError("Negative slice is not allowed in Stream")
            self._try_iter_to(item)
        elif isinstance(item, slice):
            low, high, step = item.indices(maxsize)
            if step == 0:
                raise ValueError("Step must not be 0")
            return self.__class__() << map(self.__getitem__, range(low, high, step or 1))
        else:
            raise TypeError("Invalid argument type")

        return self._cache.__getitem__(item)

    def map(self, f: Callable[[T], R]) -> Stream[R]:
        return Stream(self._iter, action_pipe=self._action_pipe + [_StreamAction(f, Stream.map)])

    def filter(self, f: Callable[[T], bool]) -> Stream[T]:
        return Stream(self._iter, action_pipe=self._action_pipe + [_StreamAction(f, Stream.filter)])

    def reduce(self, f: Callable[[T, T], T]) -> T:
        return reduce(f, self)

    def flatten(self) -> Stream[T]:
        return Stream(self._iter, action_pipe=self._action_pipe + [_StreamAction(None, Stream.flatten)])

    def flat_map(self, f: Callable[[T], AbstractSeq[R]]) -> AbstractSeq[R]:
        return self.map(f).flatten()

    def skip(self, n: int) -> Stream[T]:
        return Stream(islice(self, n, None))

    def take(self, n: int) -> Stream[T]:
        return Stream(islice(self, n))

    def to_list(self) -> List[T]:
        return [*self]

    def __str__(self):
        return f"Stream({', '.join(map(str, self._cache + ['...']))})"


class _StreamAction:

    def __init__(self, f: Optional[Callable], method: Callable):
        self.f = f
        self.method = method

    def __iter__(self):
        return iter((self.f, self.method))

    def __str__(self):
        return f"{self.method.__name__}({self.f.__name__ if self.f is not None else ''})"
