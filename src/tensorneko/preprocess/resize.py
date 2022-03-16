from typing import Tuple, Union

import torch
from cleanfid.resize import make_resizer
from einops import rearrange
from fn import F, _
from torch import Tensor, uint8, float32
from torch.nn import functional as func

from .enum import ResizeMethod, get_enum_value


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
    resize_method = get_enum_value(resize_method, ResizeMethod)

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


def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: Union[ResizeMethod, str] = ResizeMethod.BICUBIC,
    fast=False
) -> Tensor:
    """
    Resizing a video to determined size.

    Args:
        tensor (:class:`~torch.Tensor`): Video tensor (T, C, H, W)
        size ((``int``, ``int``)): Target size (H, W)
        resize_method (:class:`ResizeMethod`, optional): Resize method. Default bicubic.
        fast (bool, optional): If True, use fast mode (pytorch F.interpolate). Default False.

    Returns:
        :class:`~torch.Tensor`: The resized video.
    """
    if not fast:
        resize_method = get_enum_value(resize_method, ResizeMethod)
        image_resizer = F(resize_image, size=size, resize_method=resize_method) \
                        >> F(rearrange, pattern="c h w -> 1 c h w")
        f = F(map, image_resizer) \
            >> list \
            >> torch.vstack
        return f(tensor)
    else:
        return func.interpolate(tensor, size, mode=get_enum_value(resize_method, ResizeMethod), align_corners=False)
