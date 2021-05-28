from typing import Callable, Union, List, Tuple

from torch.nn import Module
from torch import device, Size

ModuleFactory = Union[Callable[[], Module], Callable[[int], Module]]
Device = device
Shape = Union[Size, List[int], Tuple[int, ...]]
