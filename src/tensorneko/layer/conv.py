from abc import ABC, abstractmethod
from functools import partial
from typing import Union, Optional, Callable, Tuple, TypeVar, Type, Generic

from torch import Tensor
from torch.nn import Conv1d as PtConv1d, Conv2d as PtConv2d, Conv3d as PtConv3d, Module
from torch.nn.modules.conv import _ConvNd as PtConvNd

from ..neko_module import NekoModule
from ..util import F

C = TypeVar("C", bound=PtConvNd)


class _Conv(NekoModule, ABC, Generic[C]):
    """
    An enhanced Conv version of pytorch version with combining activation and normalization.

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

        build_activation (``() -> torch.nn.Module``): An activation module builder to be used in the Conv layer.

        build_normalization (``() -> torch.nn.Module``): An normalization module builder to be used in the layer.

        normalization_after_activation (``bool``, optional): Set True then applying normalization after activation.
            Default ``False``.

    Attributes:
        conv: (:class:`~torch.nn._ConvNd`): The PyTorch ConvNd object.

        activation: (:class:`~torch.nn.Module`): The PyTorch activation module in the layer.

        normalization: (:class:`~torch.nn.Module`): The PyTorch normalization module in the layer.

    Examples::

        conv1d = tensorneko.layer.Conv1d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(3,),
            padding=(1,),
            build_activation=torch.nn.ReLU,
            build_normalization=lambda: torch.nn.BatchNorm1d(256),
            normalization_after_activation=False
        )

        conv2d = tensorneko.layer.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(3, 3),
            padding=(1, 1),
            build_activation=torch.nn.ReLU,
            build_normalization=lambda: torch.nn.BatchNorm2d(256),
            normalization_after_activation=False
        )

        conv3d = tensorneko.layer.Conv3d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            build_activation=torch.nn.ReLU,
            build_normalization=lambda: torch.nn.BatchNorm3d(256),
            normalization_after_activation=False
        )


    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1, padding: Union[int, Tuple[int, ...], str] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        device=None, dtype=None,
        build_activation: Optional[Callable[[], Module]] = None,
        build_normalization: Optional[Callable[[], Module]] = None,
        normalization_after_activation: bool = False
    ):
        super().__init__()
        self.conv: C = self._PtConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
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

    def _get_name(self) -> str:
        return self._class_name

    @property
    @abstractmethod
    def _class_name(self) -> str:
        pass

    @property
    @abstractmethod
    def _PtConv(self) -> Type[C]:
        pass


class Conv1d(_Conv[PtConv1d]):

    @property
    def _class_name(self) -> str:
        return "tensorneko.layer.Conv1d"

    @property
    def _PtConv(self) -> Type[C]:
        return PtConv1d


class Conv2d(_Conv[PtConv2d]):

    @property
    def _class_name(self) -> str:
        return "tensorneko.layer.Conv2d"

    @property
    def _PtConv(self) -> Type[C]:
        return PtConv2d


class Conv3d(_Conv[PtConv3d]):

    @property
    def _class_name(self) -> str:
        return "tensorneko.layer.Conv3d"

    @property
    def _PtConv(self) -> Type[C]:
        return PtConv3d


Conv = {
    "1d": Conv1d,
    "2d": Conv2d,
    "3d": Conv3d
}
