from torch import Tensor, log
from torch.nn import Module


class Log(Module):
    def __init__(self, eps: float = 0.):
        super().__init__()
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        return log(x) if self.eps == 0 else log(x + self.eps)
