from typing import Union, List

from torch import Tensor
from torch.nn import functional as func

from .enum import PaddingPosition, PaddingMethod, get_enum_value
from ..util import Rearrange, F


def _get_padding_pair(padding_size: int, padding_position: Union[PaddingPosition, str]) -> List[int]:
    padding_position = get_enum_value(padding_position, PaddingPosition)
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

    padding_method = get_enum_value(padding_method, PaddingMethod)
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

    padding_method = get_enum_value(padding_method, PaddingMethod)
    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        f = F(func.pad, pad=[0, 0] + pad)
    elif padding_method == "same":
        f = F() >> Rearrange("t c -> 1 c t") >> F(func.pad, pad=pad, mode="replicate") >> Rearrange("1 c t -> t c")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")

    return f(tensor)
