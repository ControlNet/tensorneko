from enum import Enum
from typing import TypeVar, Union

# Generic types
from numpy import ndarray

T = TypeVar('T')  # Any type
R = TypeVar('R')  # Return type
P = TypeVar("P", int, float, str, bool)  # Primitive type
E = TypeVar("E", bound=Enum)

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

T_CO = TypeVar("T_CO", covariant=True)  # Covariant type

# array union type
try:
    import torch
except ImportError:
    T_ARRAY = ndarray
else:
    T_ARRAY = Union[ndarray, torch.Tensor]
