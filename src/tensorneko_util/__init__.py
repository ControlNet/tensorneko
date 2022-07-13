import os.path
from os.path import dirname

from . import notebook
from . import util
from . import visualization
from . import preprocess
from . import io
from . import debug

__all__ = [
    "notebook",
    "util",
    "visualization",
    "preprocess",
    "io",
    "debug"
]

with open(os.path.join(dirname(__file__), "version.txt"), "r", encoding="UTF-8") as file:
    __version__ = file.read()
