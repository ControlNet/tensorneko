from typing import Optional

import numpy as np
from numpy import ndarray

from ..backend import VisualLib


def rgb2gray(img: ndarray, channel_first: bool = False, backend: Optional[VisualLib] = VisualLib.PIL) -> ndarray:
    """
    Convert RGB image to gray scale image.

    Args:
        img (:class:`np.ndarray`): RGB image. (H, W, 3)
        channel_first (``bool``, optional): If ``True``, the image is in channel first format. (3, H, W)
        backend (:class:`~tensorneko_util.backend.visual_lib.VisualLib`, optional): The backend library for saving.
            Default: PIL.

    Returns:
        :class:`np.ndarray`: Gray scale image. (H, W)
    """
    if channel_first:
        img = img.transpose(1, 2, 0)

    if backend == VisualLib.PIL:
        if not VisualLib.pil_available():
            raise ValueError("Pillow is not installed.")
        from PIL import Image
        return np.asarray(Image.fromarray(img).convert('L'))

    elif backend == VisualLib.OPENCV:
        if not VisualLib.opencv_available():
            raise ValueError("OpenCV is not installed.")
        import cv2
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    else:
        raise ValueError(f"Backend {backend} is not supported.")


def rgb2gray_batch(img: ndarray, channel_first: bool = False, backend: Optional[VisualLib] = VisualLib.PIL) -> ndarray:
    """
    Convert RGB images to gray scale images.

    Args:
        img (:class:`np.ndarray`): RGB images. (N, H, W, 3)
        channel_first (``bool``, optional): If ``True``, the image is in channel first format. (N, 3, H, W)
        backend (:class:`~tensorneko_util.backend.visual_lib.VisualLib`, optional): The backend library for saving.
            Default: PIL.

    Returns:
        :class:`np.ndarray`: Gray scale images. (N, H, W)
    """
    if channel_first:
        img = img.transpose(0, 2, 3, 1)

    if backend == VisualLib.PIL:
        if not VisualLib.pil_available():
            raise ValueError("Pillow is not installed.")
        from PIL import Image
        return np.asarray([np.asarray(Image.fromarray(i).convert('L')) for i in img])

    elif backend == VisualLib.OPENCV:
        if not VisualLib.opencv_available():
            raise ValueError("OpenCV is not installed.")
        import cv2
        return np.asarray([cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in img])

    else:
        raise ValueError(f"Backend {backend} is not supported.")
