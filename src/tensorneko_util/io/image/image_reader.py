from typing import Optional

import numpy as np
from einops import rearrange

from ...backend.visual_lib import VisualLib
from ...util.type import T_ARRAY
from .._default_backends import _default_image_io_backend


class ImageReader:
    """ImageReader for reading images as :class:`~torch.Tensor`"""

    @classmethod
    def of(cls, path: str, channel_first: bool = False, backend: Optional[VisualLib] = None) -> T_ARRAY:
        """
        Read image tensor of given file.

        Args:
            path (``str``): Path of the JPEG or PNG image.
            channel_first (``bool``, optional): Get image dimension (H, W, C) if False or (C, H, W) if True.
                Default: False.
            backend (:class:`~tensorneko_util.backend.visual_lib.VisualLib`, optional): The backend library for saving.
                Default: "opencv" if installed else "matplotlib".

        Returns:
            :class:`~numpy.ndarray` | :class:`~torch.Tensor`: A float tensor of image, with value range of 0. to 1.
        """
        backend = backend or _default_image_io_backend()

        if backend == VisualLib.OPENCV:
            if not VisualLib.opencv_available():
                raise ValueError("OpenCV is not installed.")
            import cv2
            img = cv2.imread(path).astype(np.float32) / 255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif backend == VisualLib.MATPLOTLIB:
            if not VisualLib.matplotlib_available():
                raise ValueError("Matplotlib is not installed.")
            import matplotlib.pyplot as plt
            img = plt.imread(path).astype(np.float32) / 255.0
        elif backend == VisualLib.PYTORCH:
            if not VisualLib.pytorch_available():
                raise ValueError("Torchvision is not installed.")
            import torchvision
            img = torchvision.io.read_image(path).float() / 255.0  # (C, H, W)
        elif backend == VisualLib.PIL:
            if not VisualLib.pil_available():
                raise ValueError("PIL is not installed.")
            import PIL.Image
            img = np.asarray(PIL.Image.open(path)).astype(np.float32) / 255.0
        else:
            raise ValueError("Unknown image reader backend: {}".format(backend))

        if channel_first:
            if backend != VisualLib.PYTORCH:
                img = rearrange(img, 'H W C -> C H W')
        else:
            if backend == VisualLib.PYTORCH:
                img = rearrange(img, 'C H W -> H W C')

        return img

    def __new__(cls, path: str, channel_first: bool = True, backend: Optional[VisualLib] = None) -> T_ARRAY:
        """Alias of :meth:`~ImageReader.of`"""
        return cls.of(path, channel_first, backend)
