from typing import Union

import torchvision.io
from numpy import ndarray
from torch import Tensor, from_numpy, uint8


class ImageWriter:

    @staticmethod
    def to_jpeg(path: str, image: Union[Tensor, ndarray], quality: int = 75):
        if type(image) == ndarray:
            image = from_numpy(image)
        torchvision.io.write_jpeg((image * 255).to(uint8), path, quality=quality)

    @staticmethod
    def to_png(path: str, image: Union[Tensor, ndarray], compression_level: int = 6):
        if type(image) == ndarray:
            image: Tensor = from_numpy(image)
        torchvision.io.write_png((image * 255).to(uint8), path, compression_level=compression_level)

    to = to_jpeg
