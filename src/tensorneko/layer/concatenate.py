from typing import Union, List, Tuple, Optional

from torch import Tensor, cat
from torch.nn import Module


class Concatenate(Module):
    def __init__(self, dim: Optional[int] = 0, out: Optional[Tensor] = None):
        super().__init__()
        self.dim = dim
        self.out = out

    def forward(self, xs: Union[List[Tensor], Tuple[Tensor, ...]]) -> Tensor:
        return cat(xs, self.dim, out=self.out)
