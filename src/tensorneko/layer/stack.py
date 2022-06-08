from typing import List, Tuple, Union

import torch
from torch import Tensor

from ..neko_module import NekoModule
from ..util import F


class Stack(NekoModule):
    """
    The module version of :meth:`torch.stack` function family.

    Args:
        mode (``str``, optional): The mode of the pytorch stack type. Default original stack.
        dim (``int``, optional): The dimension of stack apply to. Cannot use in non-default mode. Default 0.

    Examples::

        dstack = Stack("d")
        x_stack = dstack([x1, x2])

    """

    def __init__(self, mode: str = "", dim: int = 0):
        super().__init__()
        # other mode cannot specify the dim
        assert not (mode != "" and dim != 0), "Other modes cannot specify the dim"
        if mode == "":
            self.stack_func = F(torch.stack, dim=dim)
        elif mode.lower() == "d":
            self.stack_func = torch.dstack
        elif mode.lower() == "v":
            self.stack_func = torch.vstack
        elif mode.lower() == "h":
            self.stack_func = torch.hstack
        elif mode.lower() == "column":
            self.stack_func = torch.column_stack
        elif mode.lower() == "row":
            self.stack_func = torch.row_stack
        else:
            raise ValueError("""Not a valid `mode` argument. It should be in ["", "d", "v", "h", "column", "row"].""")

    def forward(self, tensors: Union[List[Tensor], Tuple[Tensor, ...]]) -> Tensor:
        return self.stack_func(tensors)
