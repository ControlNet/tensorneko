import json
from typing import overload, Union

from pandas import DataFrame


class TextWriter:
    """TextWriter for writing files for text"""

    @staticmethod
    def to_plain(path: str, text: str, encoding="UTF-8") -> None:
        """
        Save as a plain text file.

        Args:
            path (``str``): The path of output file.
            text (``str``): The content for output.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
        """
        with open(path, "w", encoding=encoding) as file:
            file.write(text)

    @staticmethod
    @overload
    def to_json(path: str, obj: DataFrame, orient: str = None) -> None:
        """
        Save as Json file from a :class:`~pandas.DataFrame`.

        Args:
            path (``str``): The path of output file.
            obj (:class:`~pandas.DataFrame`): The DataFrame used for json output.
            orient (``str``, optional): The json format option from :meth:`~pandas.DataFrame.to_json`. Default: None.
        """
        ...

    @staticmethod
    @overload
    def to_json(path: str, obj: Union[dict, list], encoding="UTF-8") -> None:
        """
        Save as Json file from a dictionary or list.

        Args:
            path (``str``): The path of output file.
            obj (``dict`` | ``list``): The json data which need to be used for output.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
        """
        ...

    @staticmethod
    def to_json(path: str, obj: Union[dict, list, DataFrame], encoding="UTF-8", orient=None) -> None:
        """
        The implementation of :meth:`~TextWriter.to_json` method.
        """
        if type(obj) in (dict, list):
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(obj))
        elif type(obj) == DataFrame:
            obj.to_json(path, orient=orient)
        else:
            raise TypeError("Not implemented type. Only support dict, list and DataFrame.")

    @staticmethod
    def to_csv(path: str, dataframe: DataFrame, index: bool = True, columns=None, header=True, sep=",") -> None:
        """
        Save as csv file from a :class:`~pandas.DataFrame`.

        Args:
            path (``str``): The path of output file.
            dataframe (:class:`~pandas.DataFrame`): The data frame used to output csv.
            index (``bool``, optional): If True then contains index in output file. Default: True
            columns (``Sequence``, optional): The columns in the data frame for output. Default: All columns.
            header (``bool``, optional): If True then contains headers in the output file. Default: True.
            sep (``str``): The string for CSV delimiter.
        """
        dataframe.to_csv(path, sep=sep, columns=columns, header=header, index=index)

    to = to_plain
