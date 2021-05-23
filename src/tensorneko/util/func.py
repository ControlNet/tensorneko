from typing import Callable, List, Dict, Iterable

from fn import F, _, Stream


def reduce_dict_by(key: str, op: Callable) -> Callable[[List[Dict[str, float]]], float]:
    return F() >> (map, _[key]) >> list >> op


def generate_inf_seq(items: Iterable) -> Stream:
    s = Stream()
    return s << items << map(_, s)
