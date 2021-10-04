from enum import Enum
from typing import Union, TypeVar, Type, List, Tuple

import torch
from torch.nn import functional as func
from cleanfid.resize import make_resizer
from einops import rearrange
from fn import F, _
from torch import Tensor, uint8, float32

from tensorneko.util import Rearrange


class ResizeMethod(Enum):
    """Resize methods can be used in :func:`~resize_image` and :func:`~resize_video`"""
    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    NEAREST = "nearest"
    LANCZOS = "lanczos"
    BOX = "box"


class PaddingMethod(Enum):
    """The padding methods used in :func:`~padding_video`"""
    ZERO = "zero"
    SAME = "same"


class PaddingPosition(Enum):
    """The padding position used in :func:`~padding_video`"""
    HEAD = "head"
    TAIL = "tail"
    AVERAGE = "average"


E = TypeVar("E", bound=Enum)


def _get_enum_value(enum: Union[E, str], EnumType: Type[E]) -> str:
    if type(enum) == EnumType:
        return enum.value
    elif type(enum) == str:
        return enum.lower()
    else:
        raise TypeError("Not matched enum type.")


def resize_image(tensor: Tensor, size: Tuple[int, int], resize_method: Union[ResizeMethod, str] = ResizeMethod.BICUBIC
) -> Tensor:
    """
    Resizing a image to determined size.

    Args:
        tensor (:class:`~torch.Tensor`): Image tensor (C, H, W)
        size ((``int``, ``int``)): Target size (H, W)
        resize_method (:class:`ResizeMethod`, optional): Resize method. Default bicubic.

    Returns:
        :class:`~torch.Tensor`: The resized image.
    """
    h, w = size
    resize_method = _get_enum_value(resize_method, ResizeMethod)

    resizer = make_resizer("PIL", True, resize_method, (w, h))
    f = F(rearrange, pattern="c h w -> h w c") \
        >> _ * 255 \
        >> (lambda x: x.to(uint8).numpy()) \
        >> resizer \
        >> torch.from_numpy \
        >> F(rearrange, pattern="h w c -> c h w") \
        >> _ / 255 \
        >> (lambda x: x.to(float32))
    return f(tensor)


def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: Union[ResizeMethod, str] = ResizeMethod.BICUBIC
) -> Tensor:
    """
    Resizing a video to determined size.

    Args:
        tensor (:class:`~torch.Tensor`): Video tensor (T, C, H, W)
        size ((``int``, ``int``)): Target size (H, W)
        resize_method (:class:`ResizeMethod`, optional): Resize method. Default bicubic.

    Returns:
        :class:`~torch.Tensor`: The resized video.
    """
    resize_method = _get_enum_value(resize_method, ResizeMethod)
    image_resizer = F(resize_image, size=size, resize_method=resize_method) \
                    >> F(rearrange, pattern="c h w -> 1 c h w")
    f = F(map, image_resizer) \
        >> list \
        >> torch.vstack
    return f(tensor)


def _get_padding_pair(padding_size: int, padding_position: Union[PaddingPosition, str]) -> List[int]:
    padding_position = _get_enum_value(padding_position, PaddingPosition)
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError("Wrong padding position. It should be zero or tail or average.")
    return pad


def padding_video(tensor: Tensor, target: int,
    padding_method: Union[PaddingMethod, str] = PaddingMethod.ZERO,
    padding_position: Union[PaddingPosition, str] = PaddingPosition.TAIL
) -> Tensor:
    """
    Padding a video in temporal.

    Args:
        tensor (:class:`~torch.Tensor`): Video tensor (T, C, H, W)
        target (``int``): The target temporal length of the padded video.
        padding_method (:class:`PaddingMethod`, optional): Padding method. Default zero padding.
        padding_position (:class:`PaddingPosition`, optional): Padding position. Default tail.

    Returns:
        :class:`~torch.Tensor`: The padded video.
    """
    t, c, h, w = tensor.shape
    padding_size = target - t

    padding_method = _get_enum_value(padding_method, PaddingMethod)
    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        f = F(func.pad, pad=[0, 0, 0, 0, 0, 0] + pad)
    elif padding_method == "same":
        f = F() >> Rearrange("t c h w -> c h w t") \
            >> F(func.pad, pad=pad + [0, 0], mode="replicate") \
            >> Rearrange("c h w t -> t c h w")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")

    return f(tensor)


def padding_audio(tensor: Tensor, target: int,
    padding_method: Union[PaddingMethod, str] = PaddingMethod.ZERO,
    padding_position: Union[PaddingPosition, str] = PaddingPosition.TAIL
) -> Tensor:
    """
    Padding an audio in temporal.

    Args:
        tensor (:class:`~torch.Tensor`): Audio tensor (T, C)
        target (``int``): The target temporal length of the padded audio.
        padding_method (:class:`PaddingMethod`, optional): Padding method. Default zero padding.
        padding_position (:class:`PaddingPosition`, optional): Padding position. Default tail.

    Returns:
        :class:`~torch.Tensor`: The padded audio.
    """
    t, c = tensor.shape
    padding_size = target - t

    padding_method = _get_enum_value(padding_method, PaddingMethod)
    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        f = F(func.pad, pad=[0, 0] + pad)
    elif padding_method == "same":
        f = F() >> Rearrange("t c -> 1 c t") >> F(func.pad, pad=pad, mode="replicate") >> Rearrange("1 c t -> t c")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")

    return f(tensor)
