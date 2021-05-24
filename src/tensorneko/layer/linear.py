from typing import Optional

from torch.nn import Linear as PtLinear
from torch.nn import Module

from tensorneko.util.type import ModuleFactory


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        build_activation: Optional[ModuleFactory] = None,
        build_normalization: Optional[ModuleFactory] = None,
        normalization_after_activation: bool = False
    ):
        super().__init__()
        self.linear = PtLinear(in_features, out_features, bias)

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()
        else:
            self.activation = None

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation
        else:
            self.normalization = None

    def forward(self, x):
        x = self.linear(x)
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                x = self.activation(x)
                x = self.normalization(x)
            else:
                x = self.normalization(x)
                x = self.activation(x)
        elif self.has_act and not self.has_norm:
            x = self.activation(x)
        elif not self.has_act and self.has_norm:
            x = self.normalization(x)
        return x
