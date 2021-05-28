from torch import Tensor
from torch.nn import Module, Sequential, ModuleList
from typing import Iterable

from ..layer import Concatenate
from ..util import ModuleFactory


class DenseBlock(Module):

    def __init__(self, sub_module_layers: Iterable[ModuleFactory], repeat=2):
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
            Concatenate(dim=1) for i in range(repeat)
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

