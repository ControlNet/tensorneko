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
def sparse2binary(x: List[int]) -> List[int]:
    return sparse2binary(np.array(x)).tolist()


@dispatch.of(list, int)
def sparse2binary(x: List[int], length: int) -> List[int]:
    return sparse2binary(np.array(x), length=length).tolist()


@dispatch
def binary2sparse(x: ndarray) -> ndarray:
    """
    Convert binary vector to sparse vector.

    Args:
        x (:class:`numpy.ndarray`): The binary vector.

    Returns:
        :class:`numpy.ndarray`: The sparse vector.

    Examples::

        x = np.array([0, 1, 1, 0, 0])
        binary2sparse(x, length=5)
        array([1, 2])
    """
    return np.where(x == 1)[0]


@dispatch.of(list)
def binary2sparse(x: List[int]) -> List[int]:
    return [i for i, v in enumerate(x) if v == 1]
