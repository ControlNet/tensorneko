from typing import Optional, Callable

from torch import Tensor
from torch.nn import Module, ModuleList

from ..util import compose


class ResidualBlock(Module):
    def __init__(self, sub_module: Module, tail_module: Optional[Module] = None):
        # x -> sub_module -> add_to(x) -> tail_module
        super().__init__()
        self.sub_module = sub_module
        self.tail_module = tail_module

    def forward(self, x: Tensor) -> Tensor:
        x = self.sub_module(x) + x
        return self.tail_module(x) if self.tail_module else x


class ResidualModule(Module):
    def __init__(self, build_block: Callable[[], ResidualBlock], repeat: int):
        super().__init__()
        self.blocks = ModuleList([build_block() for _ in range(repeat)])
        self.repeat = repeat

    def forward(self, x: Tensor) -> Tensor:
        f = compose(self.blocks)
        return f(x)
