from typing import Callable

from torch.nn import Module
from torch import device

ModuleFactory = Callable[[], Module]
Device = device