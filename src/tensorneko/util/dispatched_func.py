import torch
from torch import Tensor

from tensorneko_util.util import dispatch
from tensorneko_util.util.dispatched_func import sparse2binary as _sparse2binary
from tensorneko_util.util.dispatched_func import binary2sparse as _binary2sparse


@dispatch.base(_sparse2binary)
def sparse2binary(x: Tensor, length: int = None) -> Tensor:
    """
    Convert a sparse tensor to a binary tensor.

    Args:
        x (:class:`~torch.Tensor`): The input sparse tensor.
        length (``int``, optional): The length of the binary tensor.

    Returns:
        :class:`~torch.Tensor`: The binary tensor.
    """
    return torch.zeros(length or (x.max() + 1), dtype=torch.int).scatter_(0, x, 1)


@dispatch.base(_binary2sparse)
def binary2sparse(x: Tensor) -> Tensor:
    """
    Convert a binary tensor to a sparse tensor.

    Args:
        x (:class:`~torch.Tensor`): The input binary tensor.

    Returns:
        :class:`~torch.Tensor`: The sparse tensor.
    """
    return x.nonzero().squeeze(1)
