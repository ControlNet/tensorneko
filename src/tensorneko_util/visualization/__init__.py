from .color import Colors, ContinuousColors
from . import matplotlib
from .multi_plots import MultiPlots
from . import watcher
from . import tensorboard

__all__ = ["Colors", "ContinuousColors", "MultiPlots", "matplotlib", "watcher", "tensorboard"]

try:
    from . import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
