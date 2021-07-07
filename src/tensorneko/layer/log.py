import torch
from torch import Tensor, log

from ..neko_module import NekoModule


class Log(NekoModule):
    """
    The module version of :func:`torch.log` operation.

    Args:
        eps (``float``, optional): A bias applied to the input to avoid ``-inf``. Default ``0``.

    Examples::

        >>> log = Log()
        >>> a = torch.randn(5)
        >>> a
        tensor([ 2.3020, -0.8679, -0.2174,  2.4228, -1.2341])
        >>> log(a)
        tensor([0.8338,    nan,    nan, 0.8849,    nan])

    """

    def __init__(self, eps: float = 0.):
        super().__init__()
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        return log(x) if self.eps == 0 else log(x + self.eps)
