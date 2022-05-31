from __future__ import annotations

from abc import ABC
from typing import Generic, Callable, TYPE_CHECKING, Optional

from tensorneko_util.util.dispatcher import dispatch
from .type import P

if TYPE_CHECKING:
    from ..visualization.watcher import Component


class Ref(ABC, Generic[P]):
    _value: P

    def __init__(self, value: P):
        self._value = value
        self.bound_comp: Optional[Component] = None

    @property
    def value(self) -> P:
        return self._value

    @value.setter
    def value(self, value: P):
        self._value = value
        if self.bound_comp is not None:
            self.bound_comp.update_view()

    def apply(self, f: Callable[[P], P]) -> Ref[P]:
        new_value = f(self.value)
        assert type(new_value) is type(self.value), "The transform should be the same type."
        self.value = new_value
        return self

    def __rshift__(self, f: Callable[[P], P]) -> Ref[P]:
        return self.apply(f)


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


@dispatch.of(str)
def ref(value: str) -> StringRef:
    return StringRef(value)


@dispatch.of(int)
def ref(value: int) -> IntRef:
    return IntRef(value)


@dispatch.of(float)
def ref(value: float) -> FloatRef:
    return FloatRef(value)


@dispatch.of(bool)
def ref(value: bool) -> BoolRef:
    return BoolRef(value)
