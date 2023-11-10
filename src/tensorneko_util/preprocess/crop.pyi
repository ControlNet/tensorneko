from typing import overload, Union

from numpy import ndarray


@overload
def crop_with_padding(image: ndarray, x1: int, x2: int, y1: int, y2: int, pad_value: Union[int, float] = 0.,
    batch: bool = False
) -> ndarray: ...


@overload
def crop_with_padding(image: ndarray, x1: ndarray, x2: ndarray, y1: ndarray, y2: ndarray,
    pad_value: Union[int, float] = 0., batch: bool = False
) -> ndarray: ...
