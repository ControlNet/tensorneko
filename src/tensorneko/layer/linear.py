from typing import Optional

from fn import F
from torch import Tensor
from torch.nn import Linear as PtLinear, Dropout
from torch.nn import Module

from ..util import ModuleFactory


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        build_activation: Optional[ModuleFactory] = None,
        build_normalization: Optional[ModuleFactory] = None,
        normalization_after_activation: bool = False,
        dropout_rate: float = 0.
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

        self.has_dropout = dropout_rate > 0
        if self.has_dropout:
            self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> self.linear
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                f = f >> self.activation >> self.normalization
            else:
                f = f >> self.normalization >> self.activation
        elif self.has_act and not self.has_norm:
            f = f >> self.activation
        elif not self.has_act and self.has_norm:
            f = f >> self.normalization
        if self.has_dropout:
            f = f >> self.dropout
        return f(x)
