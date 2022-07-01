import json
from typing import Union

import pandas as pd
from pandas import DataFrame

from ...util.type import T


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
            ``dict`` | ``list`` | :class:`~pandas.DataFrame` | ``object``: The object of given json.
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

    @classmethod
    def of(cls, path: str, *args, **kwargs) -> Union[str, dict, list, DataFrame]:
        """
        Auto infer by the file extension.
        Args:
            path (``str``): Text file path.

        Returns:
            ``str`` | ``dict`` | ``list`` | :class:`~pandas.DataFrame`: Parsed value of the file.
        """
        ext = path.split(".")[-1]

        if ext == "txt":
            return cls.of_plain(path, *args, **kwargs)
        elif ext == "json":
            return cls.of_json(path, *args, **kwargs)
        elif ext == "csv":
            return cls.of_csv(path, *args, **kwargs)
        elif ext == "xml":
            return cls.of_xml(path, *args, **kwargs)
        else:
            raise Exception(f"Cannot infer the file type [{ext}]. Please explicitly invoke other reader functions.")

    def __new__(cls, path: str, *args, **kwargs) -> str:
        """Alias of :meth:`~TextReader.of_plain`"""
        return cls.of(path, *args, **kwargs)
