from .log_graph import log_graph
from .color import Colors, ContinuousColors
from . import watcher
from . import tensorboard


__all__ = [
    "log_graph",
    "watcher",
    "tensorboard",
    "Colors",
    "ContinuousColors"
]

try:
    from . import matplotlib
except ImportError:
    pass
else:
    __all__.append("matplotlib")

try:
    from . import seaborn
except ImportError:
    pass
else:
    __all__.append("seaborn")
