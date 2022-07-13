import os.path

from . import callback
from . import evaluation
from . import layer
from . import module
from . import notebook
from . import optim
from . import preprocess
from . import util
from . import visualization
from . import debug
from .neko_module import NekoModule
from .neko_model import NekoModel
from .neko_trainer import NekoTrainer
from tensorneko_util import io

__all__ = [
    "callback",
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
    "NekoModule"
]

__version__ = io.read.text(os.path.join(util.get_tensorneko_path(), "version.txt"))
