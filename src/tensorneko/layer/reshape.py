import torch
from torch import Tensor, reshape

from ..neko_module import NekoModule
from ..util import Shape


class Reshape(NekoModule):
    """
    The module version of :func:`torch.reshape` operation.

    Args:
        shape: (:class:`tensorneko.util.type.Shape`): The target shape of input tensor.

    Examples::

        >>> reshape_a = Reshape((2, 2))
        >>> a = torch.arange(4.)
        >>> reshape_a(a)
        tensor([[ 0.,  1.],
                [ 2.,  3.]])

        >>> reshape_b = Reshape((-1,))
        >>> b = torch.tensor([[0, 1], [2, 3]])
        >>> reshape_b(b)
        tensor([ 0,  1,  2,  3])

    """

    def __init__(self, shape: Shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return reshape(x, shape=self.shape)
