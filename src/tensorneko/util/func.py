from typing import Callable, List, Dict, Iterable, Sequence, Union

from fn import F, _, Stream
from fn.uniform import reduce
from torch.nn import ModuleList


def reduce_dict_by(key: str, op: Callable) -> Callable[[List[Dict[str, float]]], float]:
    return F() >> (map, _[key]) >> list >> F(reduce, op)


def summarize_dict_by(key: str, op: Callable) -> Callable[[List[Dict[str, float]]], float]:
    return F() >> (map, _[key]) >> list >> op


def generate_inf_seq(items: Iterable) -> Stream:
    s = Stream()
    return s << items << map(_, s)


def compose(fs: Union[ModuleList, Sequence[Callable]]) -> F:
    return reduce(_ >> _, fs, F())
