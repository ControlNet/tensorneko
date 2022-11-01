from . import image_browser
from . import tensorboard
from . import watcher
from .color import Colors, ContinuousColors
from ..backend import VisualLib

__all__ = [
    "watcher",
    "image_browser",
    "tensorboard",
    "Colors",
    "ContinuousColors",
]

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
