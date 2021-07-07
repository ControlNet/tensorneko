from typing import Optional, Callable, Sequence

from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

from ..neko_module import NekoModule
from ..util import compose, ModuleFactory


class ResidualBlock(NekoModule):
    """
    The ResidualBlock module is a single block of a residual connected block.
    x -> sub_module -> add_to(x) -> tail_module

    The residual connection is introduced firstly by He, Zhang, Ren and Sun (2016).

    Args:
        sub_module_layers (`Sequence[() -> Module]`): A collection of module builders for submodule route.

        tail_module_layers (`Sequence[() -> Module]`, optional): A collection of module builders for tail module which
            is placed after residual addition.

    Attributes:
        sub_module (:class:`~torch.nn.Sequential`): The PyTorch Sequential object as submodule route.

        tail_module (:class:`~torch.nn.Sequential` | ``None``): The PyTorch Sequential object as tail module after
            residual addition.

    Examples::

        block = ResidualBlock([
            lambda: torch.nn.Linear(24, 48),
            torch.nn.ReLU,
            lambda: torch.nn.Linear(48, 24)
        ])

    References:
        He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of
        the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    """

    def __init__(self, sub_module_layers: Sequence[ModuleFactory],
        tail_module_layers: Optional[Sequence[ModuleFactory]] = None
    ):

        super().__init__()
        self.sub_module = Sequential(*[build_layer() for build_layer in sub_module_layers])
        self.tail_module = None if tail_module_layers is None else \
            Sequential(*[build_layer() for build_layer in tail_module_layers])

    def forward(self, x: Tensor) -> Tensor:
        x = self.sub_module(x) + x
        return self.tail_module(x) if self.tail_module else x


class ResidualModule(NekoModule):
    """
    The Residual Module repeatedly generate Residual Blocks.

    Args:
        build_block (``() -> ResidualBlock``): The ResidualBlock builder function for repeatedly building.

        repeat (``int``): The repeat number of residual block.

    Attributes:
        blocks (:class:`~torch.nn.ModuleList`): The blocks list in the module.

    Examples::

        def build_block():
            return ResidualBlock([
                lambda: torch.nn.Linear(24, 48),
                torch.nn.ReLU,
                lambda: torch.nn.Linear(48, 24)
            ])

        residual_module = ResidualModule(build_block, repeat=10)

    """

    def __init__(self, build_block: Callable[[], ResidualBlock], repeat: int):
        super().__init__()
        self.blocks = ModuleList([build_block() for _ in range(repeat)])
        self.repeat = repeat

    def forward(self, x: Tensor) -> Tensor:
        f = compose(self.blocks)
        return f(x)
