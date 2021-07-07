from typing import Union, List, Tuple, Optional

import torch
from torch import Tensor, cat

from ..neko_module import NekoModule


class Concatenate(NekoModule):
    """
    A module version of concatenate multiple :class:`~torch.Tensor`.

    The operation used for concatenation is :func:`torch.cat`.

    Args:
        dim (``int``, optional): The dimension chosen to be concat. Default 0.
        out (:class:`~torch.Tensor`, optional): The output tensor. Default ``None``.

    Examples::

        >>> cat_dim0 = Concatenate(dim=0)
        >>> cat_dim1 = Concatenate(dim=1)
        >>> x = torch.randn(2, 3)
        >>> x
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> cat_dim0((x, x, x))
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> cat_dim1((x, x, x))
        tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
                 -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
                 -0.5790,  0.1497]])

    """

    def __init__(self, dim: Optional[int] = 0, out: Optional[Tensor] = None):
        super().__init__()
        self.dim = dim
        self.out = out

    def forward(self, xs: Union[List[Tensor], Tuple[Tensor, ...]]) -> Tensor:
        return cat(xs, self.dim, out=self.out)
