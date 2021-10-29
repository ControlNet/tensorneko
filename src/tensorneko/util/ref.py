from __future__ import annotations

from typing import Generic, Union, Tuple, Any, Iterator, Dict, overload, Optional, Mapping, Sequence, List, Iterable
import inspect
from typing_extensions import SupportsIndex

from tensorneko.util.ref import StringRef
from ..util import dispatch
from ..util.type import CT


class Ref(Generic[CT]):
    value: CT

    @dispatch
    def __new__(cls, value: str) -> StringRef:
        return StringRef(value)

    @dispatch
    def __new__(cls, value: int) -> IntRef:
        return IntRef(value)

    @dispatch
    def __new__(cls, value: float) -> FloatRef:
        return FloatRef(value)

    @dispatch
    def __new__(cls, value: bool) -> BoolRef:
        return BoolRef(value)


class StringRef(Ref[str], str):
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"StringRef({self.value})"

    def __eq__(self, other: Union[CT, Ref[CT]]) -> bool:
        return self.value == other.value if isinstance(other, Ref) else self.value == other

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other: Union[CT, Ref[CT]]) -> bool:
        return self.value < other.value if isinstance(other, Ref) else self.value < other

    def __le__(self, other: Union[CT, Ref[CT]]) -> bool:
        return self.value <= other.value if isinstance(other, Ref) else self.value <= other

    def __gt__(self, other: Union[CT, Ref[CT]]) -> bool:
        return self.value > other.value if isinstance(other, Ref) else self.value > other

    def __ge__(self, other: Union[CT, Ref[CT]]) -> bool:
        return self.value >= other.value if isinstance(other, Ref) else self.value >= other

    def __add__(self, other: Union[CT, Ref[CT]]) -> StringRef:
        return StringRef(self.value + other.value if isinstance(other, Ref) else self.value + other)

    def __mul__(self, other: Union[int, Ref[int]]) -> StringRef:
        return StringRef(self.value * other.value if isinstance(other, Ref) else self.value * other)

    def __rmul__(self, other: Union[int, Ref[int]]) -> StringRef:
        return StringRef(other.value * self.value if isinstance(other, Ref) else other * self.value)

    def capitalize(self) -> str:
        return super().capitalize()

    def casefold(self) -> str:
        return super().casefold()

    def center(self, __width: int, __fillchar: str = ...) -> str:
        return super().center(__width, __fillchar)

    def count(self, x: str, __start: Optional[SupportsIndex] = ..., __end: Optional[SupportsIndex] = ...) -> int:
        return super().count(x, __start, __end)

    def encode(self, encoding: str = ..., errors: str = ...) -> bytes:
        return super().encode(encoding, errors)

    def endswith(self, __suffix: Union[str, Tuple[str, ...]], __start: Optional[SupportsIndex] = ...,
        __end: Optional[SupportsIndex] = ...
    ) -> bool:
        return super().endswith(__suffix, __start, __end)

    def expandtabs(self, tabsize: int = ...) -> str:
        return super().expandtabs(tabsize)

    def find(self, __sub: str, __start: Optional[SupportsIndex] = ..., __end: Optional[SupportsIndex] = ...) -> int:
        return super().find(__sub, __start, __end)

    def format(self, *args: object, **kwargs: object) -> str:
        return super().format(*args, **kwargs)

    def format_map(self, map: _FormatMapMapping) -> str:
        return super().format_map(map)

    def index(self, __sub: str, __start: Optional[SupportsIndex] = ..., __end: Optional[SupportsIndex] = ...) -> int:
        return super().index(__sub, __start, __end)

    def isalnum(self) -> bool:
        return super().isalnum()

    def isalpha(self) -> bool:
        return super().isalpha()

    def isascii(self) -> bool:
        return super().isascii()

    def isdecimal(self) -> bool:
        return super().isdecimal()

    def isdigit(self) -> bool:
        return super().isdigit()

    def isidentifier(self) -> bool:
        return super().isidentifier()

    def islower(self) -> bool:
        return super().islower()

    def isnumeric(self) -> bool:
        return super().isnumeric()

    def isprintable(self) -> bool:
        return super().isprintable()

    def isspace(self) -> bool:
        return super().isspace()

    def istitle(self) -> bool:
        return super().istitle()

    def isupper(self) -> bool:
        return super().isupper()

    def join(self, __iterable: Iterable[str]) -> str:
        return super().join(__iterable)

    def ljust(self, __width: int, __fillchar: str = ...) -> str:
        return super().ljust(__width, __fillchar)

    def lower(self) -> str:
        return super().lower()

    def lstrip(self, __chars: Optional[str] = ...) -> str:
        return super().lstrip(__chars)

    def partition(self, __sep: str) -> Tuple[str, str, str]:
        return super().partition(__sep)

    def replace(self, __old: str, __new: str, __count: int = ...) -> str:
        return super().replace(__old, __new, __count)

    def removeprefix(self, __prefix: str) -> str:
        return super().removeprefix(__prefix)

    def removesuffix(self, __suffix: str) -> str:
        return super().removesuffix(__suffix)

    def rfind(self, __sub: str, __start: Optional[SupportsIndex] = ..., __end: Optional[SupportsIndex] = ...) -> int:
        return super().rfind(__sub, __start, __end)

    def rindex(self, __sub: str, __start: Optional[SupportsIndex] = ..., __end: Optional[SupportsIndex] = ...) -> int:
        return super().rindex(__sub, __start, __end)

    def rjust(self, __width: int, __fillchar: str = ...) -> str:
        return super().rjust(__width, __fillchar)

    def rpartition(self, __sep: str) -> Tuple[str, str, str]:
        return super().rpartition(__sep)

    def rsplit(self, sep: Optional[str] = ..., maxsplit: int = ...) -> List[str]:
        return super().rsplit(sep, maxsplit)

    def rstrip(self, __chars: Optional[str] = ...) -> str:
        return super().rstrip(__chars)

    def split(self, sep: Optional[str] = ..., maxsplit: int = ...) -> List[str]:
        return super().split(sep, maxsplit)

    def splitlines(self, keepends: bool = ...) -> List[str]:
        return super().splitlines(keepends)

    def startswith(self, __prefix: Union[str, Tuple[str, ...]], __start: Optional[SupportsIndex] = ...,
        __end: Optional[SupportsIndex] = ...
    ) -> bool:
        return super().startswith(__prefix, __start, __end)

    def strip(self, __chars: Optional[str] = ...) -> str:
        return super().strip(__chars)

    def swapcase(self) -> str:
        return super().swapcase()

    def title(self) -> str:
        return super().title()

    def translate(self, __table: Union[Mapping[int, Union[int, str, None]], Sequence[Union[int, str, None]]]) -> str:
        return super().translate(__table)

    def upper(self) -> str:
        return super().upper()

    def zfill(self, __width: int) -> str:
        return super().zfill(__width)

    @staticmethod
    @overload
    def maketrans(__x: Union[Dict[int, _T], Dict[str, _T], Dict[Union[str, int], _T]]) -> Dict[int, _T]: ...

    @staticmethod
    @overload
    def maketrans(__x: str, __y: str, __z: Optional[str] = ...) -> Dict[int, Union[int, None]]: ...

    @staticmethod
    def maketrans(__x: Union[Dict[int, _T], Dict[str, _T], Dict[Union[str, int], _T]]) -> Dict[int, _T]:
        return super().maketrans(__x)

    def __contains__(self, o: str) -> bool:
        return super().__contains__(o)

    def __getitem__(self, i: Union[int, slice]) -> str:
        return super().__getitem__(i)

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def __len__(self) -> int:
        return super().__len__()

    def __mod__(self, x: Any) -> str:
        return super().__mod__(x)

    def __getnewargs__(self) -> Tuple[str]:
        return super().__getnewargs__()


class IntRef(Ref[int]):
    pass


class FloatRef(Ref[float]):
    pass


class BoolRef(Ref[bool]):
    pass


ref: Ref[str] = Ref("")

ref.split()
