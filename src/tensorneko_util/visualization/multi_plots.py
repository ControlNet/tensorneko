from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import List, Optional, Callable, TypeVar

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray


class PlotType(Enum):
    IMAGE = "image"
    PLOT = "plot"


class SubPlot(ABC):
    plot_type: PlotType


@dataclass
class ImagePlot:
    image: ndarray
    title: Optional[str] = None
    kwargs: dict = field(default_factory=dict)
    plot_type = PlotType.IMAGE


@dataclass
class PlotPlot:
    plot_func: Callable[[Axes], Axes]
    subplot_kwargs: dict = field(default_factory=dict)
    plot_type = PlotType.PLOT


P = TypeVar("P", bound=SubPlot)


class MultiPlots:
    def __init__(self, n_row: Optional[int] = None, n_col: Optional[int] = None,
        figure: Optional[Figure] = None,
        title: Optional[str] = None,
        add_by_row: bool = True,
    ):
        self.n_row = n_row
        self.n_col = n_col
        self.figure = figure or plt.figure()
        self.title = title
        self.subplots: List[P] = []
        self.add_by_row = add_by_row

    def add_image(self, image: ndarray, title: Optional[str] = None, **kwargs) -> MultiPlots:
        self.subplots.append(ImagePlot(image, title, kwargs))
        return self

    def add_plot(self, plot_func: Callable[[Axes], Axes], **kwargs) -> MultiPlots:
        self.subplots.append(PlotPlot(plot_func, kwargs))
        return self

    def _determine_shape(self) -> None:
        if self.n_row is None and self.n_col is None:
            if self.add_by_row:
                self.n_row = 1
                self.n_col = len(self.subplots)
            else:
                self.n_row = len(self.subplots)
                self.n_col = 1
        elif self.n_row is None and self.n_col is not None:
            self.n_row = ceil(len(self.subplots) / self.n_col)
        elif self.n_col is None and self.n_row is not None:
            self.n_col = ceil(len(self.subplots) / self.n_row)
        else:
            if self.n_row * self.n_col < len(self.subplots):
                raise ValueError("n_row * n_col must be greater than the number of subplots")

    def _location(self, i: int) -> int:
        if self.add_by_row:
            return i + 1
        else:
            return i // self.n_col + 1 + (i % self.n_col) * self.n_row

    def plot(self) -> Figure:
        self._determine_shape()
        for i, subplot in enumerate(self.subplots):
            if subplot.plot_type == PlotType.IMAGE:
                ax = self.figure.add_subplot(self.n_row, self.n_col, self._location(i))
                ax.imshow(subplot.image, **subplot.kwargs)
                ax.set_axis_off()
                if subplot.title is not None:
                    ax.set_title(subplot.title)
            elif subplot.plot_type == PlotType.PLOT:
                ax = self.figure.add_subplot(self.n_row, self.n_col, self._location(i), **subplot.subplot_kwargs)
                ax = subplot.plot_func(ax)
        if self.title is not None:
            self.figure.suptitle(self.title)
        return self.figure
