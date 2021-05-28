from typing import Tuple

import torch
from einops import rearrange
from fn import F, _
from torch import Tensor, uint8, float32
from cleanfid.resize import make_resizer


def resize_image(tensor: Tensor, size: Tuple[int, int], filter_name: str = "bicubic") -> Tensor:
    """
    Resizing a image to determined size
    Args:
        tensor (Tensor): image tensor (C, H, W)
        size (Tuple[int, int]): target size
        filter_name (str): resize method

    Returns:
        Tensor: The resized image.
    """
    h, w = size
    resizer = make_resizer("PIL", True, filter_name, (w, h))
    f = F(rearrange, pattern="c h w -> h w c") \
        >> _ * 255 \
        >> (lambda x: x.to(uint8).numpy()) \
        >> resizer \
        >> torch.from_numpy \
        >> F(rearrange, pattern="h w c -> c h w") \
        >> _ / 255 \
        >> (lambda x: x.to(float32))
    return f(tensor)


def resize_video(tensor: Tensor, size: Tuple[int, int], filter_name: str = "bicubic"):
    """
    Resizing a video to determined size
    Args:
        tensor (Tensor): video tensor (T, C, H, W)
        size (Tuple[int, int]): target size
        filter_name (str): resize method

    Returns:
        Tensor: The resized video.
    """
    image_resizer = F(resize_image, size=size, filter_name=filter_name) >> F(rearrange, pattern="c h w -> 1 c h w")
    f = F(map, image_resizer) \
        >> list \
        >> torch.vstack
    return f(tensor)
