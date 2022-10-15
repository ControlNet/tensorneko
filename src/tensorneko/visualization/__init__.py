from .log_graph import log_graph
from tensorneko_util.visualization import Colors, ContinuousColors, tensorboard
from . import watcher
from ..backend import VisualLib

__all__ = [
    "log_graph",
    "watcher",
    "tensorboard",
    "Colors",
    "ContinuousColors",
]

if VisualLib.matplotlib_available():
    from . import matplotlib
    from tensorneko_util.visualization import MultiPlots
    __all__.extend(["matplotlib", "MultiPlots"])

try:
    from tensorneko_util.visualization.seaborn import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
