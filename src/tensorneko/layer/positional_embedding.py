from typing import Iterable

from fn import F, _
from torch import Tensor, zeros
from torch.nn import Module, Parameter, Dropout


class PositionalEmbedding(Module):

    def __init__(self, input_shape: Iterable[int], drop_rate: float = 0.5):
        super().__init__()
        self.emb = Parameter(zeros(1, *input_shape))
        self.use_dropout = drop_rate is not None and drop_rate != 0.
        if self.use_dropout:
            self.dropout = Dropout(drop_rate)


    def forward(self, x: Tensor) -> Tensor:
        f = F() >> _ + self.emb
        if self.use_dropout:
            f = f >> self.dropout
        return f(x)
