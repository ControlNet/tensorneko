import os
from os.path import dirname, abspath
from typing import Callable, List, Dict, Iterable, Sequence, Union, Any

import numpy
import numpy as np
import torch
from fn import F, _, Stream
from fn.op import identity
from fn.uniform import reduce
from torch import Tensor
from torch.nn import ModuleList, Module

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


def generate_inf_seq(items: Iterable[Any]) -> Stream:
    """
    Generate an infinity late-evaluate sequence.

    Args:
        items [``Iterable[Any]``]: Original items.

    Returns:
        :class:`~fn.stream.Stream`: The infinity sequence.

    Examples::

        >>> seq = generate_inf_seq(["a", 1, ["e0", "e1"]])
        >>> list(seq[:10])
        ['a', 1, ['e0', 'e1'], 'a', 1, ['e0', 'e1'], 'a', 1, ['e0', 'e1'], 'a']

    Note:
        The repeated reference type objects are the same object.

        >>> a_list = ["a", "b"]
        >>> seq = list(generate_inf_seq([a_list])[:3])
        >>> seq
        [['a', 'b'], ['a', 'b'], ['a', 'b']]
        >>> a_list.append("c")
        >>> seq
        [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']]

    """
    s = Stream()
    return s << items << map(_, s)


def compose(fs: Union[ModuleList, Sequence[Callable]]) -> F:
    """
    Compose functions as a pipeline function.

    Args:
        fs (``Sequence[Callable]`` | :class:`~torch.nn.ModuleList`): The functions input for composition.

    Returns:
        :class:`~fn.func.F`: The composed output function.

    Examples::

        >>> f = lambda x: x + 1
        >>> g = lambda x: x * 2
        >>> h = lambda x: x ** 2
        >>> x = 1
        >>> h(g(f(x))) == compose([f, g, h])(x)
        True

    """
    return reduce(_ >> _, fs, F())


def listdir(path: str, filter_func: Callable[[str], bool] = lambda arg: True) -> List[str]:
    """
    The full path version of :func:`os.listdir`.

    Get the all listdir result with input path.

    Args:
        path (``str``): The path for listdir.
        filter_func (``(str) -> bool``, optional): The filter function to filter the file/directory name in ``listdir``.

    Returns:
        ``List[str]``: listdir result with input path.

    Examples::

        >>> os.listdir("tensorneko/util")[:3]  # python os library
        ['configuration.py', 'func.py', 'string_getter.py']
        >>> listdir("tensorneko/util")[:3]  # tensorneko
        ['tensorneko/util/configuration.py',
         'tensorneko/util/func.py',
         'tensorneko/util/string_getter.py']

    """
    files = filter(filter_func, os.listdir(path))
    return list(map(F(os.path.join, path), files))


def with_printed(x: Any, func: Callable = identity) -> Any:
    """
    An identity function but with printed to console with some transform.

    Args:
        x (``Any``): Input.
        func (``Callable``, optional): A function used to apply the input for printing.

    Returns:
        ``Any``: Identity output.

    Examples::

        >>> x = torch.tensor([1.5, 2.5, 3.5])
        >>> y = with_printed(x, lambda tensor: tensor.dtype)
        torch.float32
        >>> x == y
        tensor([True, True, True])

    """
    print(func(x))
    return x


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


def ifelse(predicate: Callable[[Any], bool], func_true: Callable, func_false: Callable) -> Callable:
    """
    A function composition util for if-else control flow.

    Args:
        predicate (``(...) -> bool``): A predicate produce a bool.
        func_true (``(...) -> Any``): If the bool from predicate is True, return this function.
        func_false (``(...) -> Any``): If the bool from predicate is False, return this function.

    Returns:
        ``(...) -> Any``: Composed function with if-else control flow.
    """

    def wrapper(*args, **kwargs):
        if predicate(*args, **kwargs):
            return func_true(*args, **kwargs)
        else:
            return func_false(*args, **kwargs)

    return wrapper


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


def dict_add(*dicts: dict):
    """
    Merge multiple dictionaries.

    Args:
        *dicts:

    Returns:

    """
    new_dict = {}
    for each in dicts:
        new_dict.update(each)
    return new_dict


def as_list(*args, **kwargs) -> list:
    """
    Returns:
        ``list``: The list of args.

    Examples::

        >>> as_list(1, 2, 3, 4, key=5)
        [1, 2, 3, 4, 5]
    """
    return list(args) + list(kwargs.values())


def tensorneko_path() -> str:
    """
    Get the `tensorneko` library root path

    Returns:
        ``str``: The root path of `tensorneko`
    """
    return dirname(dirname(abspath(__file__)))
