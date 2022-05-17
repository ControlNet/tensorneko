from typing import Union

from einops import rearrange
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor

from tensorneko_util.visualization.matplotlib import plot_image


def imshow(tensor: Union[Tensor, ndarray], *args, **kwargs):
    """
    Use matplotlib to show image of tensor with shape (C, H, W).

    Except the first parameter, other parameters are from :func:`matplotlib.pyplot.imshow`.

    Args:
        tensor (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The image with (C, H, W) to show.

        *args: Other parameters for :func:`matplotlib.pyplot.plot`.
        **kwargs: Other parameters for :func:`matplotlib.pyplot.imshow`.

    """
    if type(tensor) is Tensor:
        tensor = tensor.detach().cpu().numpy()
    tensor = rearrange(tensor, "c h w -> h w c")
    plt.imshow(tensor, *args, **kwargs)
    plt.show()


def plot(tensor: Union[Tensor, ndarray], *args, **kwargs):
    """
    Use matplotlib to plot tensor with shape (N, 2). First column is X and second column is Y.

    Except the first parameter, other parameters are from :func:`matplotlib.pyplot.plot`.

    Args:
        tensor (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The tensor (N, 2) to plot.

        *args: Other parameters for :func:`matplotlib.pyplot.plot`.
        **kwargs: Other parameters for :func:`matplotlib.pyplot.plot`.

    """
    if type(tensor) is Tensor:
        tensor = tensor.detach().cpu().numpy()
    plt.plot(tensor[:, 0], tensor[:, 1], *args, **kwargs)
    plt.show()


plot_image = plot_image
