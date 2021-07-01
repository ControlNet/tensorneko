from torch import Tensor
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class ImageReader:
    """ImageReader for reading images as :class:`~torch.Tensor`"""

    @staticmethod
    def of(path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> Tensor:
        """
        Read image tensor of given file.

        Args:
            path (``str``): Path of the JPEG or PNG image.
            mode (:class:`~torchvision.io.image.ImageReadMode`, optional):
                the read mode used for optionally converting the image.
                Default: :class:`~torchvision.io.image.ImageReadMode.UNCHANGED`.
                See :class:`~torchvision.io.image.ImageReadMode` class for more information on various
                available modes.

        Returns:
            :class:`~torch.Tensor`: A float tensor of image (C, H, W), with value range of 0. to 1.
        """
        return read_image(path, mode) / 255
