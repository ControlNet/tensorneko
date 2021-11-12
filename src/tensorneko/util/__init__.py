from einops.layers.torch import Rearrange as _Rearrange

from .func import reduce_dict_by, summarize_dict_by, generate_inf_seq, compose, listdir, with_printed, \
    with_printed_shape, is_bad_num, ifelse, dict_add, count_parameters, as_list, identity
from .func import tensorneko_path as get_tensorneko_path
from .type import ModuleFactory, Shape, Device
from .configuration import Configuration
from .string_getter import get_activation, get_loss
from .reproducibility import Seed
from .pipe import Args as __
from .server import AbstractServer
from .dispatcher import dispatch
from .average_meter import AverageMeter

from fn.underscore import shortcut
from fn import func

Rearrange = _Rearrange
_ = shortcut
F = func.F

tensorneko_path = get_tensorneko_path()

__all__ = [
    "reduce_dict_by",
    "summarize_dict_by",
    "generate_inf_seq",
    "compose",
    "listdir",
    "with_printed",
    "with_printed_shape",
    "is_bad_num",
    "ifelse",
    "dict_add",
    "count_parameters",
    "as_list",
    "ModuleFactory",
    "Shape",
    "Device",
    "Configuration",
    "get_activation",
    "get_loss",
    "Seed",
    "__",
    "Rearrange",
    "_",
    "F",
    "tensorneko_path",
    "AbstractServer",
    "dispatch",
    "AverageMeter"
]
