import json
from typing import overload, Union

from pandas import DataFrame


class TextWriter:

    @staticmethod
    def to_plain(path: str, text: str, encoding="UTF-8"):
        with open(path, "w", encoding=encoding) as file:
            file.write(text)

    @staticmethod
    @overload
    def to_json(path: str, obj: DataFrame, encoding="UTF-8"):
        ...

    @staticmethod
    @overload
    def to_json(path: str, obj: Union[dict, list], orient: str):
        ...

    @staticmethod
    def to_json(path: str, obj: Union[dict, list, DataFrame], encoding="UTF-8", orient=None):
        if type(obj) in (dict, list):
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(obj))
        elif type(obj) == DataFrame:
            obj.to_json(path, orient=orient)
        else:
            raise TypeError("Not implemented type. Only support dict, list and DataFrame.")

    @staticmethod
    def to_csv(path: str, dataframe: DataFrame, index: bool = True, columns=None, header=None, sep=","):
        dataframe.to_csv(path, sep=sep, columns=columns, header=header, index=index)

    to = to_plain
