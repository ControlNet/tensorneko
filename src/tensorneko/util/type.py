from typing import Callable

from torch.nn import Module

ModuleFactory = Callable[[], Module]
