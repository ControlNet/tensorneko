from typing import Optional, Callable, Sequence

from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

from ..util import compose, ModuleFactory


class ResidualBlock(Module):

    def __init__(self, sub_module_layers: Sequence[ModuleFactory],
        tail_module_layers: Optional[Sequence[ModuleFactory]] = None
    ):
        # x -> sub_module -> add_to(x) -> tail_module
        super().__init__()
        self.sub_module = Sequential(*[build_layer() for build_layer in sub_module_layers])
        self.tail_module = None if tail_module_layers is None else \
            Sequential(*[build_layer() for build_layer in tail_module_layers])

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
