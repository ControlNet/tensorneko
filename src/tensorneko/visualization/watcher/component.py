import os.path
from abc import ABC, abstractmethod
from typing import Generic, List, Union

from numpy import ndarray
from torch import Tensor

from .view import View
from ...io import write
from ...util.type import T


class Component(ABC, Generic[T]):
    name: str
    views: List[View]
    _value: T

    def __init__(self, name: str, value: T):
        self.name = name
        self._value = value
        self.views = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value_in: T):
        self._value = value_in
        for view in self.views:
            view.update()

    def set(self, value_in: T):
        self.value = value_in

    @abstractmethod
    def to_dict(self):
        ...

    def update(self):
        pass

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.to_dict()}>"

    def __repr__(self):
        return self.__str__()


class Variable(Component[T]):
    """
    The components containing a value.

    Args:
        name (``str``): The name of the variable, which should be unique in
            a :class:`~tensorneko.visualization.watcher.view.View`.
        value (``T``): The wrapped value for watching.

    Attributes:
        views (``List`` [:class:`~tensorneko.visualization.web.view.View`]): The views that the variable belongs to.

    Examples::

        # create a Variable component
        var = tensorneko.visualization.watcher.Variable("x", 5)
        # update value
        var.value = 10
        # equivalent to
        var.set(15)

    """

    def __init__(self, name: str, value: T):
        super().__init__(name, value)

    def to_dict(self):
        return {
            "type": "Variable",
            "name": self.name,
            "value": str(self.value)
        }


class ProgressBar(Component[int]):
    """
    The components as a progress bar.

    Args:
        name (``str``):
            The name of the variable, which should be unique in a :class:`~tensorneko.visualization.watcher.view.View`.
        value (``int``):
            The current progress of the progress bar.
        total (``int``):
            The maximum number of the progress bar.

    Attributes:
        views (``List`` [:class:`~tensorneko.visualization.web.view.View`]): The views that the variable belongs to.

    Examples::

        # create a ProgressBar component
        pb = tensorneko.visualization.watcher.ProgressBar("process", 0, 10000)
        # update value. these 4 ways below are all equivalent.
        pb.value += 1
        pb.value = pb.value + 1
        pb.set(pb.value + 1)
        pb.add(1)

    """
    total: int

    def __init__(self, name: str, total: int, value: int = 0):
        super().__init__(name, value)
        self.total = total

    def add(self, value_in: int):
        self.value += value_in

    def to_dict(self) -> dict:
        return {
            "type": "ProgressBar",
            "name": self.name,
            "value": self.value,
            "total": self.total
        }


class Image(Component[Union[ndarray, Tensor, None]]):
    """
    The component of image for display.

    Args:
        name (``str``): The name of the image for display, which should be unique in
            a :class:`~tensorneko.visualization.watcher.view.View`.

        value (:class:`~numpy.ndarray` | :class:`~torch.Tensor` | ``None``, optional): The image array with (C, H, W).
            Value range are between [0, 1].

    Attributes:
        path (``str``): The image path related to the :class:`~tensorneko.visualization.watcher.view.View` root dir.

    Examples::

        # create a Image component
        img_comp = tensorneko.visualization.watcher.Image("img0", img_arr)
        # update image. these 2 ways below are all equivalent.
        img_comp.value = new_img_arr
        img_comp.set(new_img_arr)

    """

    def __init__(self, name: str, value: Union[ndarray, Tensor, None] = None):
        super().__init__(name, value)
        self.path = os.path.join("img", self.name)
        self._ver = 0

    def to_dict(self):
        self.update()
        return {
            "type": "Image",
            "name": self.name,
            "value": f"/img/{self.name}-{self._ver}.jpg"
        }

    def update(self):
        if self.value is not None:
            for view in self.views:
                img_dir = os.path.join(view.name, "img")
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)

                # remove old image
                pre_img_path = os.path.join(view.name, self.path) + f"-{self._ver}.jpg"
                if os.path.exists(pre_img_path):
                    os.remove(pre_img_path)

                # save new image
                self._ver += 1
                img_path = os.path.join(view.name, self.path) + f"-{self._ver}.jpg"
                # C, H, W
                write.image.to_jpeg(img_path, self.value)


class Logger(Component[List[str]]):
    """
    The component of log texts for display.

    Args:
        name (``str``): The name of the image for display, which should be unique in
            a :class:`~tensorneko.visualization.watcher.view.View`.

        value (``List[str]``, optional): The initial logs. Default [].

    Examples::

        # create a Logger component
        logger = tensorneko.visualization.watcher.Logger("training_log")
        # log new message
        logger.log("Epoch 1, loss: 0.1234, val_loss: 0.2345")

    """
    def __init__(self, name: str, value: List[str] = None):
        value = value or []
        super().__init__(name, value)

    def log(self, msg):
        self._value.append(msg)
        for view in self.views:
            view.update()

    def to_dict(self):
        return {
            "type": "Logger",
            "name": self.name,
            "value": self.value
        }
