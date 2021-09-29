import os.path

from . import callback
from . import io
from . import layer
from . import module
from . import notebook
from . import optim
from . import preprocess
from . import util
from . import visualization
from .neko_module import NekoModule
from .neko_model import NekoModel
from .neko_trainer import NekoTrainer

__all__ = [
    "callback",
    "io",
    "layer",
    "module",
    "notebook",
    "optim",
    "preprocess",
    "util",
    "visualization",
    "NekoModel",
    "NekoTrainer",
    "NekoModule"
]

__version__ = io.read.text.of(os.path.join(util.get_tensorneko_path(), "version.txt"))
