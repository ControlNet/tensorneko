from .color import Colors, ContinuousColors
from . import matplotlib
from .multi_plots import MultiPlots

__all__ = ["Colors", "ContinuousColors", "MultiPlots", "matplotlib"]

try:
    from . import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
