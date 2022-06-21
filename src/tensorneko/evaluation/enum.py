from enum import Enum


class Reduction(Enum):
    """
    Reduction method.
    """
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"
