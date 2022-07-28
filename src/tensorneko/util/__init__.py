from einops.layers.torch import Rearrange as _Rearrange

from tensorneko_util.util import AbstractServer, dispatch, AverageMeter, tensorneko_util_path
from tensorneko_util.util.fp import Seq, AbstractSeq, curry, Some, Empty, F, Stream, return_option, Option, _, __
from tensorneko_util.util import ref, Timer, Singleton
from tensorneko_util.util.eventbus import Event, EventBus, EventHandler, subscribe, subscribe_async, \
    subscribe_process, subscribe_thread
from . import type
from .configuration import Configuration
from .func import reduce_dict_by, summarize_dict_by, with_printed_shape, is_bad_num, count_parameters, compose, \
    generate_inf_seq, listdir, with_printed, ifelse, dict_add, as_list, identity, list_to_dict
from .func import get_tensorneko_path
from .reproducibility import Seed
from .string_getter import get_activation, get_loss
from .type import ModuleFactory, Shape, Device

Rearrange = _Rearrange

tensorneko_path = get_tensorneko_path()
tensorneko_util_path = tensorneko_util_path

__all__ = [
    "reduce_dict_by",
    "summarize_dict_by",
    "identity",
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
    "list_to_dict",
    "ref",
    "ModuleFactory",
    "Shape",
    "Device",
    "Configuration",
    "get_activation",
    "get_loss",
    "Seed",
    "__",
    "Stream",
    "Seq",
    "return_option",
    "Option",
    "Some",
    "Empty",
    "Rearrange",
    "_",
    "F",
    "tensorneko_path",
    "tensorneko_util_path",
    "AbstractServer",
    "dispatch",
    "AverageMeter",
    "AbstractSeq",
    "curry",
    "Timer",
    "Event",
    "EventBus",
    "EventHandler",
    "subscribe",
    "subscribe_async",
    "subscribe_process",
    "subscribe_thread",
    "Singleton",
]
