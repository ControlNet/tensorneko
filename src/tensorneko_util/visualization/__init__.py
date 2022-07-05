from .color import Colors, ContinuousColors
from . import matplotlib
from .multi_plots import MultiPlots
from . import watcher
from .tensorboard import Server

__all__ = ["Colors", "ContinuousColors", "MultiPlots", "matplotlib", "watcher", "Server"]

try:
    from . import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
