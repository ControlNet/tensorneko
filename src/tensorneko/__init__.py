import os.path

from . import backend
from . import dataset
from . import debug
from . import evaluation
from . import io
from . import layer
from . import module
from . import notebook
from . import optim
from . import preprocess
from . import util
from . import visualization
from .io import read, write
from .neko_module import NekoModule

__version__ = io.read.text(os.path.join(util.get_tensorneko_path(), "version.txt"))

__all__ = [
    "dataset",
    "backend",
    "evaluation",
    "io",
    "layer",
    "module",
    "notebook",
    "optim",
    "preprocess",
    "util",
    "visualization",
    "debug",
    "NekoModule",
    "read",
    "write",
]

try:
    from . import callback
except ImportError:
    pass
else:
    __all__.append("callback")

try:
    from .neko_model import NekoModel
except ImportError:
    pass
else:
    __all__.append("NekoModel")

try:
    from .neko_trainer import NekoTrainer
except ImportError:
    pass
else:
    __all__.append("NekoTrainer")
