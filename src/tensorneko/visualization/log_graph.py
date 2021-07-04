from typing import Iterable

import torch
from torch import randn
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from ..util import Shape


def log_graph(module: Module, input_shape_list: Iterable[Shape], log_dir: str = None) -> None:
    """
    Plot a graph of model for TensorBoard.

    Args:
        module (:class:`~torch.nn.Module`): The module you want to plot the structure.
        input_shape_list (``Iterable`` [:class:`~tensorneko.util.type.Shape`]): The shapes of input variables.
        log_dir (``str``, optional): The log directory for save. Default "None".

    Examples:

        Make a structure graph of a module of :class:`tensorneko.layer.Linear`.

        .. code-block:: python

            linear = tensorneko.layer.Linear(
                in_features=128,
                out_features=256,
                build_activation=torch.nn.LeakyReLU,
                build_normalization=lambda: torch.nn.BatchNorm1d(256),
                dropout_rate=0.5
            )
            log_graph(linear, [(128,)], log_dir="graphs")

        Then, run the TensorBoard server in terminal. And check the graph in TensorBoard.

        .. code-block:: bash

            tensorboard --logdir="graphs"

    """

    writer = SummaryWriter(log_dir=log_dir)
    xs = list(map(lambda shape: randn(torch.Size((1, *shape))), input_shape_list))
    if len(xs) == 1:
        writer.add_graph(module, xs[0])
    else:
        writer.add_graph(module, xs)
    writer.close()
