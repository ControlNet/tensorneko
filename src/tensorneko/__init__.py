import os.path

from . import backend
from . import callback
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
from .neko_model import NekoModel
from .neko_module import NekoModule
from .neko_trainer import NekoTrainer

__version__ = io.read.text(os.path.join(util.get_tensorneko_path(), "version.txt"))

__all__ = [
    "callback",
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
    "NekoModel",
    "NekoTrainer",
    "NekoModule",
    "read",
    "write",
]


