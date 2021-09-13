from typing import Union

import torchvision.io
from numpy import ndarray
from torch import Tensor, from_numpy, uint8


class ImageWriter:
    """ImageWriter for saving images from :class:`~torch.Tensor` or :class:`~numpy.ndarray`"""

    @staticmethod
    def to_jpeg(path: str, image: Union[Tensor, ndarray], quality: int = 75) -> None:
        """
        Save as jpeg files from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, H, W).

        Args:
            path (``str``): The path of output file.
            image (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The image tensor for output.
            quality (``int``, optional): The quality level for :func:`torchvision.io.write_jpeg`. Default: 75.
        """
        if type(image) == ndarray:
            image = from_numpy(image)
        torchvision.io.write_jpeg((image * 255).to(uint8), path, quality=quality)

    @staticmethod
    def to_png(path: str, image: Union[Tensor, ndarray], compression_level: int = 6) -> None:
        """
        Save as png files from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, H, W).

        Args:
            path (``str``): The path of output file.
            image (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The image tensor for output.
            compression_level (``int``, optional): The compression level for :func:`torchvision.io.write_png`. Default: 6
        """
        if type(image) == ndarray:
            image: Tensor = from_numpy(image)
        torchvision.io.write_png((image * 255).to(uint8), path, compression_level=compression_level)

    to = to_jpeg
