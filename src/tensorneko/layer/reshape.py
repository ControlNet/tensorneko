from torch import Tensor, reshape
from torch.nn import Module

from ..util import Shape


class Reshape(Module):
    def __init__(self, shape: Shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return reshape(x, shape=self.shape)
