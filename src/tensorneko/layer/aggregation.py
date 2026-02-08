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
        func_map = {
            "mean": torch.mean,
            "sum": torch.sum,
            "max": torch.max,
            "min": torch.min,
        }
        if mode not in func_map:
            raise ValueError("Wrong mode value. It should be in [mean, sum, max, min]")
        self.agg_func = F(func_map[mode]) if dim is None else F(func_map[mode], dim=dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.agg_func(x)
