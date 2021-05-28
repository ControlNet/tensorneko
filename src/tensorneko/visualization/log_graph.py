from typing import Iterable

import torch
from torch import randn
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


def log_graph(model: Module, input_shape_list: Iterable[Iterable[int]], log_dir: str = None):
    writer = SummaryWriter(log_dir=log_dir)
    xs = list(map(lambda shape: randn(torch.Size((1, *shape))), input_shape_list))
    if len(xs) == 1:
        writer.add_graph(model, xs[1])
    else:
        writer.add_graph(model, xs)
    writer.close()
