from .color import Colors, ContinuousColors

__all__ = ["Colors", "ContinuousColors"]

try:
    from . import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
