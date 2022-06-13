from abc import abstractmethod
from typing import Union, Tuple, Type

import torch
from torch import Tensor
from torch.nn import Conv2d

from .conv import _Conv, C


class _MaskedConv2d(Conv2d):
    """
    Naive masked 2D convolution layer from PixelCNN. There are 2 types (A and B). For the details, please check the
    original paper.

    For gated masked convolution, please refer to :class:`~tensorneko.module.gated_conv.GatedConv`.

    Args:
        in_channels (``int``): Number of channels in the input image

        out_channels (``int``): Number of channels produced by the convolution

        kernel_size (``int`` | ``(int, ...)``): Size of the convolving kernel

        stride (``int`` | ``(int, ...)``, optional): Stride of the convolution. Default: 1

        padding (``int`` | ``(int, ...)`` | ``str``, optional): Padding added to all four sides of
            the input. Default: 0

        dilation (``int`` | ``(int, ...)``, optional): Spacing between kernel elements. Default: 1

        groups (``int``, optional): Number of blocked connections from input
            channels to output channels. Default: 1

        bias (``bool``, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

        padding_mode (``string``, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

        device: The device to place the layer passed to pytorch Conv layer.

        dtype: The data type of the layer passed to pytorch Conv layer.

    References:

        van den Oord, A., Kalchbrenner, N., Espeholt, L., kavukcuoglu,  koray, Vinyals, O., & Graves, A. (2016).
        Conditional Image Generation with PixelCNN Decoders. Advances in Neural Information Processing Systems, 29.
        https://proceedings.neurips.cc/paper/2016/hash/b1301141feffabac455e1f90a7de2054-Abstract.html

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1, padding: Union[int, Tuple[int, ...], str] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        device=None, dtype=None
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)
        self.register_buffer("mask", self.make_mask())

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)

    @abstractmethod
    def make_mask(self) -> Tensor:
        pass


class _MaskedConv2dA(_MaskedConv2d):

    def make_mask(self) -> Tensor:
        mask = torch.ones(self.weight.data.shape)
        _, _, kernel_h, kernel_w = self.weight.size()
        mask[:, :, kernel_h // 2, kernel_w // 2:] = 0
        mask[:, :, kernel_h // 2 + 1:] = 0
        return mask


class _MaskedConv2dB(_MaskedConv2d):

    def make_mask(self) -> Tensor:
        mask = torch.ones(self.weight.data.shape)
        _, _, kernel_h, kernel_w = self.weight.size()
        mask[:, :, kernel_h // 2, kernel_w // 2 + 1:] = 0
        mask[:, :, kernel_h // 2 + 1:] = 0
        return mask


class MaskedConv2dA(_Conv[_MaskedConv2dA]):

    @property
    def _class_name(self) -> str:
        return "tensorneko.layer.MaskedConv2dA"

    @property
    def _PtConv(self) -> Type[C]:
        return _MaskedConv2dA


class MaskedConv2dB(_Conv[_MaskedConv2dB]):

    @property
    def _class_name(self) -> str:
        return "tensorneko.layer.MaskedConv2dB"

    @property
    def _PtConv(self) -> Type[C]:
        return _MaskedConv2dB


MaskedConv2d = {
    "A": MaskedConv2dA,
    "B": MaskedConv2dB
}


class _VerticalStackConv2dA(_MaskedConv2d):

    def make_mask(self) -> Tensor:
        mask = torch.ones(self.weight.data.shape)
        _, _, kernel_h, kernel_w = self.weight.size()
        print(self.weight.size())
        mask[:, :, kernel_h // 2:] = 0
        return mask


class _VerticalStackConv2dB(_MaskedConv2d):

    def make_mask(self) -> Tensor:
        mask = torch.ones(self.weight.data.shape)
        _, _, kernel_h, kernel_w = self.weight.size()
        mask[:, :, kernel_h // 2 + 1:] = 0
        return mask


class _HorizontalStackConv2dA(_MaskedConv2d):

    def make_mask(self) -> Tensor:
        mask = torch.ones(self.weight.data.shape)
        _, _, kernel_h, kernel_w = self.weight.size()
        mask[:, :, :, kernel_w // 2:] = 0
        return mask


class _HorizontalStackConv2dB(_MaskedConv2d):

    def make_mask(self) -> Tensor:
        mask = torch.ones(self.weight.data.shape)
        _, _, kernel_h, kernel_w = self.weight.size()
        mask[:, :, :, kernel_w // 2 + 1:] = 0
        return mask
