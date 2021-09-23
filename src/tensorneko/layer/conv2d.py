from typing import Union, Optional, Callable, Tuple

from fn import F
from torch import Tensor
from torch.nn import Conv2d as PtConv2d, Module

from ..neko_module import NekoModule


class Conv2d(NekoModule):
    """
    An enhanced Conv2d version of :class:`torch.nn.Conv2d` with combining activation and normalization.

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

        build_activation (``() -> torch.nn.Module``): An activation module builder to be used in the Conv2d layer.

        build_normalization (``() -> torch.nn.Module``): An normalization module builder to be used in the layer.

        normalization_after_activation (``bool``, optional): Set True then applying normalization after activation.
            Default ``False``.

    Attributes:
        conv2d: (:class:`~torch.nn.Conv2d`): The PyTorch Conv2d object.

        activation: (:class:`~torch.nn.Module`): The PyTorch activation module in the layer.

        normalization: (:class:`~torch.nn.Module`): The PyTorch normalization module in the layer.

    Examples::

        conv2d = tensorneko.layer.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=(1, 1),
            build_activation=torch.nn.ReLU,
            build_normalization=lambda: torch.nn.BatchNorm2d(256),
            normalization_after_activation=False
        )

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1, padding: Union[int, Tuple[int, ...], str] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
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
