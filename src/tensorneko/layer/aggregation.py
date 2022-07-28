from typing import Sequence, Union

import torch
from torch import Tensor

from ..neko_module import NekoModule
from ..util import F


class Aggregation(NekoModule):
    """
    The torch module for aggregation.

    Args:
        mode (``str``, optional): The mode of aggregation. Default "mean".
        dim (``int`` | ``Sequence[int]``, optional): The dimension chosen to apply aggregate function. Default None.

    Examples::

        x = torch.rand(1, 16, 32, 32)
        global_avg_pooling = Aggregation("avg", dim=(1, 2, 3))
        x_pooled = max_pooling(x)

    """
    def __init__(self, mode: str = "mean", dim: Union[int, Sequence[int]] = None):
        super().__init__()
        if mode == "mean":
            self.agg_func = F(torch.mean, dim=dim)
        elif mode == "sum":
            self.agg_func = F(torch.sum, dim=dim)
        elif mode == "max":
            self.agg_func = F(torch.max, dim=dim)
        elif mode == "min":
            self.agg_func = F(torch.min, dim=dim)
        else:
            raise ValueError("Wrong mode value. It should be in [mean, sum, max, min]")

    def forward(self, x: Tensor) -> Tensor:
        return self.agg_func(x)
