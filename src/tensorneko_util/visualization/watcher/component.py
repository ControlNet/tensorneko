from __future__ import annotations

import os.path
from abc import ABC, abstractmethod
from typing import Generic, List, Union, Dict, TYPE_CHECKING, Any, Optional

from einops import rearrange

from ...backend import VisualLib
from ...util.ref import Ref
from ...util.type import T, P, T_ARRAY

if TYPE_CHECKING:
    from .view import View


class Component(ABC, Generic[T]):
    name: str
    views: List[View]
    _value: T
    components: Dict[str, Component] = {}

    def __init__(self, name: str, value: T):
        self.name = name
        self._value = value
        self.views = []
        Component.components[self.name] = self

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value_in: T) -> None:
        self._value = value_in
        self.update_view()

    def update_view(self):
        for view in self.views:
            view.update()

    def set(self, value_in: T) -> None:
        self.value = value_in

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...

    def update(self) -> None:
        pass

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {self.to_dict()}>"

    def __repr__(self) -> str:
        return self.__str__()

    def bind(self, ref: Ref[P]) -> Bindable[P]:
        ...


class Bindable(ABC, Generic[P]):
    ref: Ref[P]

    @property
    def value(self):
        return self.ref.value

    @value.setter
    def value(self, value_in: P):
        raise ValueError("Cannot set value of bindable component")


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Variable",
            "name": self.name,
            "value": str(self.value)
        }

    def bind(self, ref: Ref[P]) -> BindableVariable[P]:
        return BindableVariable(ref, self.name)


class BindableVariable(Bindable[P], Variable[P]):

    def __init__(self, ref: Ref[P], name: str):
        super().__init__(name, ref.value)
        Bindable.__init__(self)
        self.ref = ref
        self.ref.bound_comp = self


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

    def add(self, value_in: int) -> None:
        self.value += value_in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ProgressBar",
            "name": self.name,
            "value": self.value,
            "total": self.total
        }

    def bind(self, ref: Ref[P]) -> BindableProgressBar[P]:
        return BindableProgressBar(ref, self.name, self.total)


class BindableProgressBar(Bindable[int], ProgressBar):

    def __init__(self, ref: Ref[P], name: str, total: int):
        super().__init__(name, total, ref.value)
        Bindable.__init__(self)
        self.ref = ref
        self.ref.bound_comp = self


class Image(Component[Optional[T_ARRAY]]):
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

        # create an Image component
        img_comp = tensorneko.visualization.watcher.Image("img0", img_arr)
        # update image. these 2 ways below are all equivalent.
        img_comp.value = new_img_arr
        img_comp.set(new_img_arr)

    """

    def __init__(self, name: str, value: Optional[T_ARRAY] = None):
        super().__init__(name, value)
        self.path = os.path.join("img", self.name)
        self._ver = 0
        if VisualLib.matplotlib_available():
            import matplotlib.pyplot as plt
            self._plt = plt
        else:
            raise ImportError("matplotlib is required for Image component.")

    def to_dict(self) -> Dict[str, Any]:
        self.update()
        return {
            "type": "Image",
            "name": self.name,
            "value": f"/img/{self.name}-{self._ver}.jpg"
        }

    def update(self) -> None:
        if self.value is not None:
            for view in self.views:
                img_dir = os.path.join("watcher", view.name, "img")
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)

                # remove old image
                pre_img_path = os.path.join("watcher", view.name, self.path) + f"-{self._ver}.jpg"
                if os.path.exists(pre_img_path):
                    os.remove(pre_img_path)

                # save new image
                self._ver += 1
                img_path = os.path.join("watcher", view.name, self.path) + f"-{self._ver}.jpg"
                # C, H, W
                image = rearrange(self.value, "c h w -> h w c")
                self._plt.imsave(img_path, image)


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

    def log(self, msg: str) -> None:
        self._value.append(msg)
        self.update_view()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Logger",
            "name": self.name,
            "value": self.value
        }


class LineChart(Component[List[Dict[str, Union[float, str]]]]):
    """
    The component for logging a line chart.
    """

    def __init__(self, name: str, value: Optional[List[Dict[str, Union[float, str]]]] = None,
        x_label: str = "index", y_label: str = "value"
    ):
        value = value or []
        super().__init__(name, value)
        self.x_label = x_label
        self.y_label = y_label

    def add(self, x: float, y: float, label: str = "") -> None:
        self._value.append({"x": x, "y": y, "label": label})
        for view in self.views:
            view.update()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LineChart",
            "name": self.name,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "value": self.value
        }
