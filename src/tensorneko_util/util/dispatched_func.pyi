from typing import overload, List

from numpy import ndarray


@overload
def sparse2binary(x: ndarray, length: int = None) -> ndarray:
    ...


@overload
def sparse2binary(x: List[int], length: int = None) -> ndarray:
    ...
