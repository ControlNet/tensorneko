from typing import Union, Sequence, Optional, Callable

from fn import F
from torch import Tensor
from torch.nn import Conv2d as PtConv2d
from torch.nn import Module


class Conv2d(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence],
        stride: Union[int, Sequence] = 1, padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        build_activation: Optional[Callable[[], Module]] = None,
        build_normalization: Optional[Callable[[], Module]] = None,
        normalization_after_activation: bool = False
    ):
        super().__init__()

        self.conv2d = PtConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
            padding_mode
        )

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> self.conv2d
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                f = f >> self.activation >> self.normalization
            else:
                f = f >> self.normalization >> self.activation
        elif self.has_act and not self.has_norm:
            f = f >> self.activation
        elif not self.has_act and self.has_norm:
            f = f >> self.normalization
        return f(x)
