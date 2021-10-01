import json
from dataclasses import fields, dataclass
from typing import Union, Dict, Any

import pandas as pd
from pandas import DataFrame

from tensorneko.util.type import T


class TextReader:
    """TextReader for reading text file"""

    @staticmethod
    def of_plain(path: str, encoding: str = "UTF-8") -> str:
        """
        Read texts of a file.

        Args:
            path (``str``): Text file path.
            encoding (``str``, optional): File encoding. Default "UTF-8".

        Returns:
            ``str``: The texts of given file.
        """
        with open(path, "r", encoding=encoding) as file:
            text = file.read()
        return text

    @staticmethod
    def of_json(path: str, to_df: bool = False, orient: str = None, cls: T = None, encoding: str = "UTF-8"
    ) -> Union[T, dict, list, DataFrame]:
        """
        Read json files to :class:`~pandas.DataFrame` or ``list`` or ``dict``.

        Args:
            path (``str``): Json file path.
            to_df (``bool``, optional):
                True then return :class:`~pandas.DataFrame`, else return ``list`` or ``dict``. Default: False
            orient (``str``, optional): The json format parameter from :func:`pandas.read_json`. Default: None
            cls: (``T``, optional): The object of the json read for. The type should be decorated by
                :func:`json_data`. This should be ``T`` or ``List[T]`` or ``List[List[T]]``
            encoding (``str``, optional): The encoding for python ``open`` function. Default: "UTF-8"

        Returns:
            ``dict`` | ``list`` | :class:`~pandas.DataFrame`: The object of given json.
        """

        if to_df:
            return pd.read_json(path, orient=orient)
        else:
            if cls is not None:
                if "typing.List[" in str(cls):
                    with open(path, "r", encoding=encoding) as file:
                        obj: list = json.loads(file.read())

                    inner_type = cls.__args__[0]
                    if "typing.List[" in str(inner_type):
                        inner_inner_type = inner_type.__args__[0]
                        try:
                            is_json_data = inner_type.is_json_data
                        except AttributeError:
                            is_json_data = False
                        if is_json_data:
                            obj = [[inner_inner_type(e) for e in each] for each in obj]

                    else:
                        try:
                            is_json_data = inner_type.is_json_data
                        except AttributeError:
                            is_json_data = False

                        if is_json_data:
                            obj = list(map(inner_type, obj))
                else:
                    with open(path, "r", encoding=encoding) as file:
                        obj = cls(json.loads(file.read()))
            else:
                with open(path, "r", encoding=encoding) as file:
                    obj = json.loads(file.read())

            return obj

    of_csv = pd.read_csv
    of_xml = pd.read_xml
    of = of_plain


def json_data(cls):
    """
    The decorator used for JSON parser.

    Examples::

        @json_data
        class Point:
            x: int
            y: int

        # Use for tensorneko read
        point = tensorneko.io.read.text.of_json("path/to/file.json", cls=Point)

        # Use for the Python json loads
        with open("path/to/file.json", "r") as f:
            point = json.loads(f.read(), object_hook=Point)

        print(point.x, point.y)

    """

    def constructor(self, d: Dict[str, Any]):
        for field in fields(self.__class__):
            k = field.name
            t = field.type
            if "typing.List[" in str(t):
                inner_type = t.__args__[0]
                if "typing.List[" in str(inner_type):
                    inner_inner_type = inner_type.__args__[0]
                    try:
                        is_json_data = inner_inner_type.is_json_data
                    except AttributeError:
                        is_json_data = False
                    if is_json_data:
                        v = [[inner_inner_type(e) for e in each] for each in d[k]]
                    else:
                        v = d[k]
                else:
                    try:
                        is_json_data = inner_type.is_json_data
                    except AttributeError:
                        is_json_data = False

                    if is_json_data:
                        v = list(map(inner_type, d[k]))
                    else:
                        v = d[k]
            else:
                try:
                    is_json_data = t.is_json_data
                except AttributeError:
                    is_json_data = False

                if is_json_data:
                    v = t(d[k])
                else:
                    v = d[k]

            self.__dict__[k] = v

    processed_class: type = dataclass(cls)
    processed_class.__init__ = constructor
    processed_class.is_json_data = True
    return processed_class
