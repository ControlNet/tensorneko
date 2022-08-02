from __future__ import annotations

from typing import Generic, Dict, Optional, Union, Tuple

from ..util.type import K, V


class BiMap(Generic[K, V]):
    def __init__(self, items: Dict[K, V] = None):
        self.forward: Dict[K, V] = {}
        self.backward: Dict[V, K] = {}
        if items is not None:
            for key, value in items.items():
                self.forward[key] = value
                self.backward[value] = key

    def __setitem__(self, key: K, value: V):
        self.forward[key] = value
        self.backward[value] = key

    def __getitem__(self, key: K) -> V:
        return self.forward[key]

    def get(self, key: K, default: Optional[V] = None) -> V:
        if default is None:
            return self.forward[key]
        else:
            return self.forward.get(key, default)

    def get_key(self, value: V, default: Optional[K] = None) -> K:
        if default is None:
            return self.backward[value]
        else:
            return self.backward.get(value, default)

    def __len__(self):
        return len(self.forward)

    def __contains__(self, key: K) -> bool:
        return key in self.forward

    def __delitem__(self, key: K):
        del self.backward[self.forward[key]]
        del self.forward[key]

    def clear(self):
        self.forward.clear()
        self.backward.clear()

    def __copy__(self):
        bimap = BiMap()
        bimap.forward = self.forward.copy()
        bimap.backward = self.backward.copy()

    def items(self):
        return self.forward.items()

    def keys(self):
        return self.forward.keys()

    def values(self):
        return self.forward.values()

    def __iter__(self):
        return self.forward.__iter__()

    def update(self, other: Union[Dict[K, V], BiMap[K, V]]):
        if isinstance(other, BiMap):
            self.forward.update(other.forward)
            self.backward.update(other.backward)
        else:
            self.forward.update(other)
            self.backward.update({v: k for k, v in other.items()})

    def pop(self, key: K) -> V:
        value = self.forward.pop(key)
        self.backward.pop(value)
        return value

    def popitem(self) -> Tuple[K, V]:
        key, value = self.forward.popitem()
        self.backward.pop(value)
        return key, value

    def __repr__(self):
        return f"BiMap({self.forward})"

    def __str__(self):
        return f"BiMap({self.forward})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BiMap):
            return self.forward == other.forward and self.backward == other.backward
        else:
            return False

    def __hash__(self):
        return hash(self.forward) ^ hash(self.backward)

    def __bool__(self):
        return bool(self.forward)
