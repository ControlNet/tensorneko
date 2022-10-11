from . import tensorboard
from . import watcher
from .color import Colors, ContinuousColors
from ..backend import VisualLib

__all__ = ["Colors", "ContinuousColors", "watcher", "tensorboard"]

if VisualLib.matplotlib_available():
    from . import matplotlib
    from .multi_plots import MultiPlots
    __all__.extend(["matplotlib", "MultiPlots"])

try:
    from . import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
