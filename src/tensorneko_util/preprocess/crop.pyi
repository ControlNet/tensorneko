from typing import overload, Union

from numpy import ndarray


@overload
def crop_with_padding(image: ndarray, x1: int, x2: int, y1: int, y2: int, pad_value: Union[int, float] = 0.,
    batch: bool = False
) -> ndarray: ...
