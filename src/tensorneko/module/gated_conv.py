from typing import Tuple

from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F

from ..neko_module import NekoModule
from ..layer.masked_conv2d import _VerticalStackConv2dA, _VerticalStackConv2dB, \
    _HorizontalStackConv2dA, _HorizontalStackConv2dB


class GatedConv(NekoModule):

    def __init__(self, mask_type: str, in_channels: int, kernel_size: int, residual: bool = True, **kwargs):
        super().__init__()
        assert mask_type in ["A", "B"]
        out_channels = in_channels * 2
        self.residual = residual
        self.Conv = {
            "vertical": _VerticalStackConv2dA if mask_type == "A" else _VerticalStackConv2dB,
            "horizontal": _HorizontalStackConv2dA if mask_type == "A" else _HorizontalStackConv2dB
        }

        self.conv_vertical = self.Conv["vertical"](in_channels, out_channels, kernel_size, **kwargs)
        self.conv_horizontal = self.Conv["horizontal"](in_channels, out_channels, kernel_size, **kwargs)
        self.vertical_to_horizontal = Conv2d(out_channels, out_channels, kernel_size=1)
        self.horizontal_proj = Conv2d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def activation(x: Tensor) -> Tensor:
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)

    def forward(self, x_v: Tensor, x_h: Tensor) -> Tuple[Tensor, Tensor]:
        # vertical stack
        v_feat = self.conv_vertical(x_v)
        v_out = self.activation(v_feat)

        # horizontal stack
        h_feat = self.conv_horizontal(x_h) + self.vertical_to_horizontal(v_feat)
        h_feat = self.activation(h_feat)
        h_out = self.horizontal_proj(h_feat)

        # residual connection
        if self.residual:
            h_out = h_out + x_h

        return h_out, v_out
