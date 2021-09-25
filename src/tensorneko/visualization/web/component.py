from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List

from .view import View
from ...util.type import T


@dataclass
class Component(ABC):
    name: str
    views: List[View] = field(repr=False, init=False, default_factory=list)

    @abstractmethod
    def to_dict(self):
        ...


@dataclass
class Variable(Component, Generic[T]):
    """
    The components containing a value

    Args:
        name (``str``): The name of the variable, which should be unique in
            a :class:`~tensorneko.visualization.web.view.View`.
        __value (``T``): The wrapped value for watching.

    Attributes:
        views (``List`` [:class:`~tensorneko.visualization.web.view.View`]): The views that the variable belongs to.

    Examples::

        # create a Variable component
        var = tensorneko.visualization.web.Variable("x", 5)
        # update value
        var.value = 10
        # equivalent to
        var.set(15)

    """

    __value: T

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value_in: T):
        self.__value = value_in
        for view in self.views:
            view.update()

    def set(self, value_in: T):
        self.value = value_in

    def to_dict(self):
        return {
            "type": "Variable",
            "name": self.name,
            "value": str(self.value)
        }


@dataclass
class ProgressBar(Component):
    """
    The components as a progress bar.

    Args:
        name (``str``):
            The name of the variable, which should be unique in a :class:`~tensorneko.visualization.web.view.View`.
        __value (``int``):
            The current progress of the progress bar.
        total (``int``):
            The maximum number of the progress bar.

    Attributes:
        views (``List`` [:class:`~tensorneko.visualization.web.view.View`]): The views that the variable belongs to.

    Examples::

        # create a ProgressBar component
        pb = tensorneko.visualization.web.ProgressBar("process", 0, 10000)
        # update value. these 4 ways below are all equivalent.
        pb.value += 1
        pb.value = pb.value + 1
        pb.set(pb.value + 1)
        pb.add(1)

    """

    __value: int
    total: int

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value_in: int):
        self.__value = value_in
        for view in self.views:
            view.update()

    def set(self, value_in: int):
        self.value = value_in

    def add(self, value_in: int):
        self.value += value_in

    def to_dict(self) -> dict:
        return {
            "type": "ProgressBar",
            "name": self.name,
            "value": self.value,
            "total": self.total
        }
