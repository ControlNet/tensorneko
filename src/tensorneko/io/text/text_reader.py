import json
from typing import Union

import pandas as pd
from pandas import DataFrame


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
    def of_json(path: str, to_df: bool = False, orient: str = None, encoding: str = "UTF-8"
    ) -> Union[dict, list, DataFrame]:
        """
        Read json files to :class:`~pandas.DataFrame` or ``list`` or ``dict``.

        Args:
            path (``str``): Json file path.
            to_df (``bool``, optional):
                True then return :class:`~pandas.DataFrame`, else return ``list`` or ``dict``. Default: False
            orient (``str``, optional): The json format parameter from :func:`pandas.read_json`. Default: None
            encoding (``str``, optional): The encoding for python ``open`` function. Default: "UTF-8"

        Returns:
            ``dict`` | ``list`` | :class:`~pandas.DataFrame`: The object of given json.
        """

        if to_df:
            return pd.read_json(path, orient=orient)
        else:
            with open(path, "r", encoding=encoding) as file:
                obj = json.loads(file.read())
            return obj

    of_csv = pd.read_csv
    of_xml = pd.read_xml
    of = of_plain
