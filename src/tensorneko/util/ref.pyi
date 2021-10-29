from __future__ import annotations

from typing import Generic, overload

from ..util.type import P


class Ref(Generic[P]):
    value: P
    @overload
    def __new__(cls, value: str) -> StringRef: ...
    @overload
    def __new__(cls, value: int) -> IntRef: ...
    @overload
    def __new__(cls, value: float) -> FloatRef: ...
    @overload
    def __new__(cls, value: bool) -> BoolRef: ...

class StringRef(Ref[str]):
    def __str__(self) -> str: ...

class IntRef(Ref[int]):
    def __int__(self) -> int: ...

class FloatRef(Ref[float]):
    def __float__(self) -> float: ...

class BoolRef(Ref[bool]):
    def __bool__(self) -> bool: ...
