from typing import Union

import numpy as np
from numpy import ndarray

from tensorneko_util.util import dispatch


@dispatch
def crop_with_padding(image: ndarray, x1: int, x2: int, y1: int, y2: int, pad_value: Union[int, float] = 0.,
    batch: bool = False
) -> ndarray:
    """
    Crop image with padding.

    This function is implemented for :class:`~numpy.ndarray`.

    Args:
        image (:class:`~numpy.ndarray`): Image to be cropped. The shape should be ([B,] H, W) or ([B,] H, W, C).
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

    if not batch:
        image = image[np.newaxis, ...]

    crop_shape = np.array([y2 - y1, x2 - x1])

    if len(image.shape) == 3:
        b, h, w = image.shape
        cropped = np.full((b, *crop_shape), pad_value, dtype=image.dtype)
    elif len(image.shape) == 4:
        b, h, w, c = image.shape
        cropped = np.full((b, *crop_shape, c), pad_value, dtype=image.dtype)
    else:
        raise ValueError("Invalid shape, the image should be one of following shapes: ([B,] H, W) or ([B,] H, W, C)")

    # compute cropped index of image
    image_y_start, image_x_start = np.clip([y1, x1], 0, [h, w])
    image_y_end, image_x_end = np.clip([y2, x2], 0, [h, w])

    # compute target index of output
    crop_y_start, crop_x_start = np.clip([-y1, -x1], 0, crop_shape)
    crop_y_end, crop_x_end = crop_shape - np.clip([y2 - h, x2 - w], 0, crop_shape)

    # assign values
    cropped[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end] = \
        image[:, image_y_start:image_y_end, image_x_start:image_x_end]

    return cropped if batch else cropped[0]


@dispatch
def crop_with_padding(image: ndarray, x1: np.int32, x2: np.int32, y1: np.int32, y2: np.int32,
    pad_value: Union[int, float] = 0., batch: bool = False
) -> ndarray:
    return crop_with_padding(image, int(x1), int(x2), int(y1), int(y2), pad_value, batch)
