import pathlib
from typing import Optional

import numpy as np
from einops import rearrange
from numpy import ndarray

from .._default_backends import _default_image_io_backend
from ...backend.visual_lib import VisualLib
from ...util.type import T_ARRAY


class ImageWriter:
    """ImageWriter for saving images from :class:`~torch.Tensor` or :class:`~numpy.ndarray`"""

    @classmethod
    def _convert_img_format(cls, img_in: T_ARRAY, backend: VisualLib, channel_first: bool):
        # torchvision requires (C, H, W) but others requires (H, W, C), normalize the shape
        if backend == VisualLib.PYTORCH and not channel_first:
            img_in = rearrange(img_in, 'H W C -> C H W')
        elif backend != VisualLib.PYTORCH and channel_first:
            img_in = rearrange(img_in, 'C H W -> H W C')

        # convert value range to [0, 255] int
        if isinstance(img_in, ndarray):
            img_out = (img_in * 255).astype(np.uint8)
        else:
            try:
                import torch
                if isinstance(img_in, torch.Tensor):
                    img_out = (img_in * 255).type(torch.IntTensor)
                else:
                    raise ValueError("Unknown data type. The image array type must be numpy.ndarray or torch.Tensor.")
            except ImportError:
                raise ValueError("Unknown data type. The image array type must be numpy.ndarray or torch.Tensor.")

        return img_out

    @classmethod
    def to_jpeg(cls, path: str, image: T_ARRAY, quality: int = 75, channel_first: bool = False,
        backend: Optional[VisualLib] = None
    ) -> None:
        """
        Save as jpeg files from :class:`~torch.Tensor` or :class:`~numpy.ndarray`. The value range is [0, 1].

        Args:
            path (``str``): The path of output file.
            image (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The image tensor for output.
            quality (``int``, optional): The quality level for jpg from 0 to 100. Higher means quality is better.
                Default: 75.
            channel_first (``bool``, optional): The flag for channel first.
                Specify the image shape is (H, W, C) if False, else (C, H, W) if True Default: False.
            backend (:class:`~tensorneko_util.backend.visual_lib.VisualLib`, optional): The backend library for saving.
                Default: "opencv" if installed else "matplotlib".
        """
        backend = backend or _default_image_io_backend()
        image = cls._convert_img_format(image, backend, channel_first)

        if backend == VisualLib.OPENCV:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif backend == VisualLib.MATPLOTLIB:
            import matplotlib.pyplot as plt
            plt.imsave(path, image, format='jpeg')
        elif backend == VisualLib.PIL:
            import PIL.Image
            PIL.Image.fromarray(image).save(path, format='jpeg', quality=quality)
        elif backend == VisualLib.PYTORCH:
            import torchvision.io
            torchvision.io.write_jpeg(image, path, quality=quality)
        else:
            raise ValueError("Unknown backend library.")

    @classmethod
    def to_png(cls, path: str, image: T_ARRAY, compression_level: int = 6, channel_first: bool = False,
        backend: Optional[VisualLib] = None
    ) -> None:
        """
        Save as png files from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, H, W).

        Args:
            path (``str``): The path of output file.
            image (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The image tensor for output.
            compression_level (``int``, optional): The compression level png from 0 to 9. Higher means quality is worse.
                Default: 6
            channel_first (``bool``, optional): The flag for channel first.
                Specify the image shape is (H, W, C) if False, else (C, H, W) if True Default: False.
            backend (:class:`~tensorneko_util.backend.visual_lib.VisualLib`, optional): The backend library for saving.
                Default: "opencv" if installed else "matplotlib".
        """
        backend = backend or _default_image_io_backend()
        image = cls._convert_img_format(image, backend, channel_first)
        if backend == VisualLib.OPENCV:
            if not VisualLib.opencv_available():
                raise ValueError("OpenCV is not installed.")
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        elif backend == VisualLib.MATPLOTLIB:
            if not VisualLib.matplotlib_available():
                raise ValueError("Matplotlib is not installed.")
            import matplotlib.pyplot as plt
            plt.imsave(path, image, format='png')
        elif backend == VisualLib.PIL:
            if not VisualLib.pil_available():
                raise ValueError("PIL is not installed.")
            import PIL.Image
            PIL.Image.fromarray(image).save(path, format='png', compress_level=compression_level)
        elif backend == VisualLib.PYTORCH:
            if not VisualLib.pytorch_available():
                raise ValueError("Torchvision is not installed.")
            import torchvision.io
            torchvision.io.write_png(image, path, compression_level=compression_level)
        else:
            raise ValueError("Unknown backend library.")

    @classmethod
    def to(cls, path: str, image: T_ARRAY, *args, **kwargs) -> None:
        """
        Save as png files from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, H, W).

        Args:
            path (``str``): The path of output file.
            image (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The image tensor for output.
            *args: The arguments for :meth:`ImageWriter.to_jpeg` or :meth:`ImageWriter.to_png`.
            **kwargs: The keyword arguments for :meth:`ImageWriter.to_jpeg` or :meth:`ImageWriter.to_png`.
        """
        ext = pathlib.Path(path).suffix
        if ext in ('.jpg', '.jpeg'):
            cls.to_jpeg(path, image, *args, **kwargs)
        elif ext == '.png':
            cls.to_png(path, image, *args, **kwargs)
        else:
            raise ValueError("Unknown file extension. Now only support .jpg, .jpeg and .png.")

    def __new__(cls, path: str, image: T_ARRAY, *args, **kwargs) -> None:
        """Alias of :meth:`ImageWriter.to`"""
        cls.to(path, image, *args, **kwargs)
