from functools import reduce
from os.path import dirname, abspath
from typing import Callable, List, Dict, Sequence, Union

import numpy
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from tensorneko_util.util import F, _
from tensorneko_util.util.func import generate_inf_seq, listdir, with_printed, ifelse, dict_add, as_list, \
    identity, list_to_dict, compose
from .type import T, A


def reduce_dict_by(key: str, op: Callable[[T, T], T]) -> Callable[[List[Dict[str, T]]], T]:
    """
    Apply reducer function to a list of dictionaries.

    Args:
        key (``str``): The key of the value in dicts
        op (``(T, T) -> T``): A reduce operation function used to reduce the values.

    Returns:
        ``List[Dict[str, T]] -> T``: The reducer aggregation function for the list of dictionaries.

    Examples::

        >>> x = [
        >>>     {"a": 1, "b": 2, "c": torch.Tensor([3])},
        >>>     {"a": 3, "b": 4, "c": torch.Tensor([5])},
        >>>     {"a": 2.3, "b": -1, "c": torch.Tensor([0])}
        >>> ]
        >>> reduce_dict_by("a", lambda x1, x2: x1 + x2)(x)
        6.3
        >>> reduce_dict_by("b", lambda x1, x2: x1 - x2)(x)
        -1
        >>> reduce_dict_by("c", lambda x1, x2: 10 * x1 + x2)(x)
        tensor([350.])

    """
    return F() >> (map, _[key]) >> list >> F(reduce, op)


def summarize_dict_by(key: str, op: Callable[[Union[Sequence[T], T]], T]
) -> Callable[[List[Dict[str, T]]], T]:
    """
    Apply summarizer function to a list of dictionaries.

    Args:
        key (``str``): The key of the value in dicts
        op (``(Sequence[T] | T) -> T``):
            A summarizer operation function used to summarize value from a sequence.

    Returns:
        ``List[Dict[str, T]] -> T``:
            The summarizer aggregation function for the list of dictionaries.

    Examples::

        >>> x = [
        >>>    {"a": 1, "b": torch.Tensor([2]), "c": 3, "d": np.array([1.5])},
        >>>    {"a": 3, "b": torch.Tensor([4]), "c": 5, "d": np.array([2.5])},
        >>>    {"a": 2.3, "b": torch.Tensor([-1]), "c": 0, "d": np.array([-1.0])}
        >>> ]
        >>> summarize_dict_by("a", sum)(x)
        6.3
        >>> summarize_dict_by("b", torch.mean)(x)
        tensor(1.6667)
        >>> def some_func(x):
        >>>     x = list(map(str, x))
        >>>     x = "".join(x)
        >>>     x = int(x)
        >>>     return x
        >>> summarize_dict_by("c", some_func)(x)
        350
        >>> summarize_dict_by("d", lambda x: np.sum(x, axis=0))(x)
        array([3.])

    """
    return F() >> (map, _[key]) >> list >> ifelse(
        lambda xs: type(xs[0]) is Tensor,
        func_true=torch.vstack,
        func_false=ifelse(
            lambda xs: type(xs[0]) is np.ndarray,
            func_true=numpy.vstack,
            func_false=_
        )
    ) >> op


def with_printed_shape(x: A) -> A:
    """
    An identity function but with shape printed .

    Args:
        x (:class:`~torch.Tensor` | :class:`numpy.ndarray`): Input.

    Returns:
        :class:`~torch.Tensor` | :class:`numpy.ndarray`: Identity output.

    Examples::

        >>> x = torch.tensor([1.5, 2.5, 3.5])
        >>> y = with_printed_shape(x)
        torch.Size([3])
        >>> x == y
        tensor([True, True, True])
    """
    return F(with_printed, func=lambda tensor: tensor.shape)(x)


def is_bad_num(x: Tensor) -> Tensor:
    """
    Checking the element in input tensor is bad number (NaN or Inf).

    Args:
        x (:class:`~torch.Tensor`): Input tensor.

    Returns:
        :class:`~torch.Tensor`: Checking result for bad numbers.
    """
    return torch.logical_or(torch.isnan(x), torch.isinf(x))


def count_parameters(module: Module) -> int:
    """
    Counts the number of parameters of a :class:`~torch.nn.Module`.

    Args:
        module (:class:`~torch.nn.Module`): The input module for counting parameters.

    Returns:
        ``int``: Number of parameters.
    """
    return sum(p.numel() for p in module.parameters())


def get_tensorneko_path() -> str:
    """
    Get the `tensorneko` library root path

    Returns:
        ``str``: The root path of `tensorneko`
    """
    return dirname(dirname(abspath(__file__)))


# merge package namespace
compose = compose
generate_inf_seq = generate_inf_seq
listdir = listdir
with_printed = with_printed
ifelse = ifelse
dict_add = dict_add
as_list = as_list
identity = identity
list_to_dict = list_to_dict
