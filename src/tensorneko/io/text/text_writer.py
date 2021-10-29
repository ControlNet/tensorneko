import json
from typing import Union

from pandas import DataFrame

from ...util import dispatch


class TextWriter:
    """TextWriter for writing files for text"""

    @staticmethod
    def to_plain(path: str, text: str, encoding: str = "UTF-8") -> None:
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
    @dispatch
    def to_json(path: str, obj: DataFrame, orient: str = None) -> None:
        """
        Save as Json file from a :class:`~pandas.DataFrame`.

        Args:
            path (``str``): The path of output file.
            obj (:class:`~pandas.DataFrame`): The DataFrame used for json output.
            orient (``str``, optional): The json format option from :meth:`~pandas.DataFrame.to_json`. Default: None.
        """
        obj.to_json(path, orient=orient)

    @staticmethod
    @dispatch
    def to_json(path: str, obj: Union[dict, list, object], encoding: str = "UTF-8") -> None:
        """
        Save as Json file from a dictionary, list or json_dict.

        Args:
            path (``str``): The path of output file.
            obj (``dict`` | ``list`` | ``object``): The json data which need to be used for output. The type should be
                ``dict``, ``list`` or the ``object`` decorated by :func:`~tensorneko.io.text.text_reader.json_data`.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
        """
        if type(obj) is dict:
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(obj))
        elif "is_json_data" in type(obj).__dict__ and type(obj).is_json_data:
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(obj.to_dict()))
        elif type(obj) is list:
            if len(obj) == 0 or type(obj[0]) in (dict, list):
                with open(path, "w", encoding=encoding) as file:
                    file.write(json.dumps(obj))
            elif "is_json_data" in type(obj[0]).__dict__ and type(obj[0]).is_json_data:
                obj = [each.to_dict() for each in obj]
                with open(path, "w", encoding=encoding) as file:
                    file.write(json.dumps(obj))
        else:
            raise TypeError("Not implemented type. Only support dict, list, json_data and DataFrame.")

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

    @staticmethod
    def to_xml(path: str, dataframe: DataFrame):
        # TODO
        dataframe.to_xml(path)

    @classmethod
    def to(cls, path: str, *args, **kwargs) -> None:
        """
        Save text files with auto inferred type.

        Args:
            path (``str``): The path of output file.
            *args: Other arguments in corresponded TextWriter functions.
            **kwargs: Other arguments in corresponded TextWriter functions.
        """
        ext = path.split(".")[-1]

        if ext == "txt":
            return cls.to_plain(path, *args, **kwargs)
        elif ext == "json":
            return cls.to_json(path, *args, **kwargs)
        elif ext == "csv":
            return cls.to_csv(path, *args, **kwargs)
        elif ext == "xml":
            return cls.to_xml(path, *args, **kwargs)
        else:
            raise Exception(f"Cannot infer the file type [{ext}]. Please explicitly invoke other writer functions.")
