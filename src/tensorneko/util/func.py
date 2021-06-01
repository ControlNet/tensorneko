import os
from typing import Callable, List, Dict, Iterable, Sequence, Union, Any

import numpy
import numpy as np
import torch
from fn import F, _, Stream
from fn.op import identity
from fn.uniform import reduce
from torch import Tensor
from torch.nn import ModuleList, Module


def reduce_dict_by(key: str, op: Callable) -> Callable[[List[Dict[str, float]]], Any]:
    return F() >> (map, _[key]) >> list >> F(reduce, op)


def summarize_dict_by(key: str, op: Callable) -> Callable[[List[Dict[str, float]]], Any]:
    return F() >> (map, _[key]) >> list >> ifelse(
        lambda xs: type(xs[0]) is Tensor,
        func_true=torch.vstack,
        func_false=ifelse(
            lambda xs: type(xs[0]) is np.ndarray,
            func_true=numpy.vstack,
            func_false=_
        )
    ) >> op


def generate_inf_seq(items: Iterable) -> Stream:
    s = Stream()
    return s << items << map(_, s)


def compose(fs: Union[ModuleList, Sequence[Callable]]) -> F:
    return reduce(_ >> _, fs, F())


# full path version of os.listdir
def listdir(path):
    files = filter(_ != ".DS_Store", os.listdir(path))
    return list(map(F(os.path.join, path), files))


def with_printed(x, func=identity):
    print(func(x))
    return x


def with_printed_shape(x):
    return F(with_printed, func=lambda tensor: tensor.shape)(x)


def ifelse(predicate: Callable[[any], bool], func_true: Callable, func_false: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if predicate(*args, **kwargs):
            return func_true(*args, **kwargs)
        else:
            return func_false(*args, **kwargs)

    return wrapper


def is_bad_num(x: Tensor) -> Tensor:
    return torch.logical_or(torch.isnan(x), torch.isinf(x))


def count_parameters(model: Module):
    return sum(p.numel() for p in model.parameters())
