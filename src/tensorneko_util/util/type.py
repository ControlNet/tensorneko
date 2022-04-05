from enum import Enum
from typing import TypeVar


# Generic types
T = TypeVar('T')  # Any type
R = TypeVar('R')  # Return type
P = TypeVar("P", int, float, str, bool)  # Primitive type
E = TypeVar("E", bound=Enum)

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
