from typing import Iterable

from torch import Tensor
from torch.nn import Sequential, ModuleList

from ..layer import Concatenate
from ..neko_module import NekoModule
from ..util import ModuleFactory


class DenseBlock(NekoModule):
    """
    The DenseBlock can be used to build a block with repeatable submodules with dense connections. This structure is
    proposed by Huang, Liu, Van Der Maaten, & Weinberger (2017).

    Args:
        sub_module_layers (``List`` [``(int) -> torch.nn.Module``]):
            A collection of module factory builder to build a "layer" in DenseBlock. In the DenseBlock, there will be a
            submodule generated repeatedly for several times. The factory function takes a repeat_index as input, and
            build a :class:`~torch.nn.Module`.

        repeat (``int``): Number of repeats for each layer in DenseBlock.

    Attributes:
        build_sub_module (``(int) -> torch.nn.Module``): The module factory function to build a submodule in DenseBlock.

        sub_modules (:class:`~torch.nn.ModuleList`): The ModuleList of all submodules.

        concatenates (:class:`~torch.nn.ModuleList`): The ModuleList of all Concatenate layers in DenseBlock.

    Examples::

        # batch norm builder
        def build_bn(i=0):
            return BatchNorm2d(2 ** i * self.c)

        # conv2d builder
        def build_conv2d_1x1(i=0):
            return Conv2d(2 ** i * self.c, 2 ** i * self.c * 4, (1, 1), build_activation=ReLU,
                build_normalization=lambda: BatchNorm2d(2 ** i * self.c * 4))

        # conv
        def build_conv2d_3x3(i=0):
            return Conv2d(2 ** i * self.c * 4, 2 ** i * self.c, (3, 3), padding=(1, 1))

        dense_block = tensorneko.module.DenseBlock((
            build_bn,
            lambda i: ReLU(),
            build_conv2d_1x1,
            build_conv2d_3x3
        ), repeat=4)

    References:
        Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).

    """

    def __init__(self, sub_module_layers: Iterable[ModuleFactory], repeat: int = 2):
        super().__init__()

        def build_sub_module(i):
            return Sequential(*[
                build_layer(i) for build_layer in sub_module_layers
            ])

        self.build_sub_module = build_sub_module
        self.sub_modules = ModuleList([
            self.build_sub_module(i) for i in range(repeat)
        ])
        self.concatenates = ModuleList([
            Concatenate(dim=1) for _ in range(repeat)
        ])

    def forward(self, x: Tensor) -> Tensor:
        xs = []
        for i in range(len(self.sub_modules)):
            # concat with previous output
            xs.append(x)
            if i != 0:
                x = self.concatenates[i - 1](xs)
            # forward with submodule
            x = self.sub_modules[i](x)

        xs.append(x)
        x = self.concatenates[-1](xs)
        return x
