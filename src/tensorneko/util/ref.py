from __future__ import annotations

from typing import Generic, Callable

from ..util import dispatch
from ..util.type import P


class Ref(Generic[P]):
    value: P

    def __init__(self, value: P):
        self.value = value

    @dispatch
    def __new__(cls, value: str) -> StringRef:
        return StringRef(value)

    @dispatch
    def __new__(cls, value: int) -> IntRef:
        return IntRef(value)

    @dispatch
    def __new__(cls, value: float) -> FloatRef:
        return FloatRef(value)

    @dispatch
    def __new__(cls, value: bool) -> BoolRef:
        return BoolRef(value)

    def apply(self, f: Callable[[P], P]) -> Ref[P]:
        new_value = f(self.value)
        assert type(new_value) is type(self.value), "The transform should be the same type."
        self.value = new_value
        return self

    def __rshift__(self, f: Callable[[P], P]) -> Ref[P]:
        return self.apply(f)

    def bind(self, var: "Component"):
        ...


class StringRef(Ref[str]):
    def __str__(self):
        return self.value


class IntRef(Ref[int]):
    def __int__(self):
        return self.value


class FloatRef(Ref[float]):
    def __float__(self):
        return self.value


class BoolRef(Ref[bool]):
    def __bool__(self):
        return self.value

