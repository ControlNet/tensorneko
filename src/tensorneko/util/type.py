from enum import Enum
from typing import Callable, Union, List, Tuple, TypeVar

import numpy as np
import torch
from torch import device, Size
from torch.nn import Module

from tensorneko_util.util.type import T, R, E, P, T1, T2, T3

ModuleFactory = Union[Callable[[], Module], Callable[[int], Module]]
"""The module builder type of ``() -> torch.nn.Module | (int) -> torch.nn.Module``"""


Device = device
"""Device type of :class:`torch.device`"""


Shape = Union[Size, List[int], Tuple[int, ...]]
"""Shape type. Normally is ``List[int]``, ``Tuple[int, ...]`` or :class:`torch.Size`"""


# Generic types
A = TypeVar("A", torch.Tensor, np.ndarray)  # Array type

# merge package namespace
T = T
R = R
P = P
E = E
T1 = T1
T2 = T2
T3 = T3
