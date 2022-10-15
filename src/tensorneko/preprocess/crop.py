from typing import Union

import torch
from einops import rearrange
from torch import Tensor

from tensorneko_util.util import dispatch
from tensorneko_util.preprocess.crop import crop_with_padding as _crop_with_padding


@dispatch.base(_crop_with_padding)
def crop_with_padding(image: Tensor, x1: int, x2: int, y1: int, y2: int, pad_value: Union[int, float] = 0.,
    batch: bool = False
) -> Tensor:
    """
    Crop image with padding.

    This function is implemented for :class:`~torch.Tensor`.

    Args:
        image (:class:`~torch.Tensor`): Image to be cropped. The shape should be ([B,] H, W) or ([B,] C, H, W).
        x1 (``int``): Left-top x-coordinate of the cropped area.
        x2 (``int``): Right-bottom x-coordinate of the cropped area.
        y1 (``int``): Left-top y-coordinate of the cropped area.
        y2 (``int``): Right-bottom y-coordinate of the cropped area.
        pad_value (``int`` | ``float``, optional): The value to fill the padded region. Default 0.
        batch (``bool``, optional): Indicate if the input is a batch of images. Default False.

    Returns:
        :class:`~numpy.ndarray`: Cropped image.

    """
    assert y2 > y1 and x2 > x1, "Should follow y2 > y1 and x2 > x1"

    # normalize the dimension to [B, C, H, W]
    if not batch:
        image = image.unsqueeze(0)

    crop_shape = torch.tensor([y2 - y1, x2 - x1])

    has_channel = len(image.shape) == 4

    if len(image.shape) == 3:
        image = rearrange(image, "b h w -> b 1 h w")
    elif len(image.shape) != 4:
        raise ValueError("Invalid shape, the image should be one of following shapes: ([B,] H, W) or ([B,] C, H, W)")

    b, c, h, w = image.shape
    cropped = torch.full((b, c, *crop_shape), pad_value, dtype=image.dtype)

    image_shape = torch.tensor([h, w])
    begin = torch.tensor([y1, x1])
    end = torch.tensor([y2, x2])

    # compute cropped index of image
    image_y_start, image_x_start = torch.clamp(begin, min=torch.tensor(0), max=image_shape)
    image_y_end, image_x_end = torch.clamp(end, min=torch.tensor(0), max=image_shape)

    # compute target index of output
    crop_y_start, crop_x_start = torch.clamp(-begin, min=torch.tensor(0), max=crop_shape)
    crop_y_end, crop_x_end = crop_shape - torch.clamp(end - image_shape, min=torch.tensor(0), max=crop_shape)

    # assign values
    cropped[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end] = \
        image[:, :, image_y_start:image_y_end, image_x_start:image_x_end]

    # remove channel or batch dimension if the input don't have
    if not has_channel:
        cropped = cropped.squeeze(1)  # B C H W -> B H W

    if not batch:
        cropped = cropped.squeeze(0)  # B [C] H W -> [C] H W

    return cropped
