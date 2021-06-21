from torch import Tensor
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class ImageReader:

    @staticmethod
    def of(path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> Tensor:
        return read_image(path, mode) / 255
