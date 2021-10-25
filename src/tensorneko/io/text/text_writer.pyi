from typing import overload, Union

from pandas import DataFrame


class TextWriter:

    @staticmethod
    def to_plain(path: str, text: str, encoding="UTF-8") -> None: ...

    @staticmethod
    @overload
    def to_json(path: str, obj: DataFrame, orient: str = None) -> None: ...

    @staticmethod
    @overload
    def to_json(path: str, obj: Union[dict, list, object], encoding="UTF-8") -> None: ...

    @staticmethod
    def to_csv(path: str, dataframe: DataFrame, index: bool = True, columns=None, header=True, sep=",") -> None: ...

    @staticmethod
    def to_xml(path: str, dataframe: DataFrame) -> None: ...  # TODO

    @classmethod
    def to(cls, path: str, *args, **kwargs) -> None: ...
