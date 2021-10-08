import torch
from torch import Tensor

from tensorneko import NekoModule
from tensorneko.util import F


class Aggregation(NekoModule):
    def __init__(self, mode: str = "mean", dim: int = None):
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
