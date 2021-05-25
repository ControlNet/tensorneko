from __future__ import annotations

from fn import F
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

from ..layer import Concatenate


class InceptionModule(Module):

    def __init__(self):
        super().__init__()
        self.sub_seqs = ModuleList()
        self.channel_concat = Concatenate(dim=1)

    def add_sub_sequence(self, *layers: Module) -> InceptionModule:
        self.sub_seqs.append(Sequential(*layers))
        return self

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> (map, lambda seq: seq(x)) >> list >> self.channel_concat
        return f(self.sub_seqs)

