from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray


def plot_image(image: ndarray, title: Optional[str] = None, figure: Optional[Figure] = None, **kwargs) -> Figure:
    fig = figure or plt.figure()
    ax = fig.add_subplot()
    ax.imshow(image, **kwargs)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return fig
