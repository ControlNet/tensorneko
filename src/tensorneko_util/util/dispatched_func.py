from typing import List

import numpy as np
from numpy import ndarray

from .dispatcher import dispatch


@dispatch
def sparse2binary(x: ndarray, length: int = None) -> ndarray:
    """
    Convert sparse vector to binary vector.

    Args:
        x (:class:`numpy.ndarray`): The sparse vector.
        length (``int``, optional): The length of binary vector.

    Returns:
        :class:`numpy.ndarray`: The binary vector.

    Examples::

        x = np.array([1, 2])
        sparse2binary(x, length=5)
        array([0, 1, 1, 0, 0])
    """
    if length is None:
        length = x.max() + 1
    binary = np.zeros(length, dtype=np.int32)
    binary[x] = 1
    return binary


@dispatch.of(list)
def sparse2binary(x: List[int]) -> ndarray:
    return sparse2binary(np.array(x))


@dispatch.of(list, int)
def sparse2binary(x: List[int], length: int) -> ndarray:
    return sparse2binary(np.array(x), length=length)
