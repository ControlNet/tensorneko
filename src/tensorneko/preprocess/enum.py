from enum import Enum
from typing import Union, Type

from PIL import Image

from ..util.type import E


class ResizeMethod(Enum):
    """Resize methods can be used in :func:`~resize_image` and :func:`~resize_video`"""
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    LANCZOS = Image.LANCZOS
    BOX = Image.BOX


class PaddingMethod(Enum):
    """The padding methods used in :func:`~padding_video`"""
    ZERO = "zero"
    SAME = "same"


class PaddingPosition(Enum):
    """The padding position used in :func:`~padding_video`"""
    HEAD = "head"
    TAIL = "tail"
    AVERAGE = "average"


def get_enum_value(enum: Union[E, str], EnumType: Type[E]) -> str:
    if type(enum) == EnumType:
        return enum.value
    elif type(enum) == str:
        return enum.lower()
    else:
        raise TypeError("Not matched enum type.")
