import json
import warnings
from typing import Union
from pathlib import Path

from .._path_conversion import _path2str


class JsonWriter:

    @staticmethod
    def _write_json(path: str, obj: Union[dict, list], encoding: str = "UTF-8", indent: int = 2,
        ensure_ascii: bool = False, fast: bool = True
    ) -> None:
        if not fast:
            with open(path, "w", encoding=encoding) as file:
                json.dump(obj, file, indent=indent, ensure_ascii=ensure_ascii)
        else:
            import orjson
            with open(path, "wb") as file:
                if indent == 4:
                    warnings.warn("orjson does not support indent 4, will use indent 2 instead.")
                    option = orjson.OPT_INDENT_2
                elif indent == 2:
                    option = orjson.OPT_INDENT_2
                else:
                    option = None
                file.write(orjson.dumps(obj, option=option))

    @classmethod
    def to(cls, path: Union[str, Path], obj: Union[dict, list, object], encoding: str = "UTF-8", indent: int = 2,
        ensure_ascii: bool = False, fast: bool = True
    ) -> None:
        """
        Save as Json file from a dictionary, list or json_dict.

        Args:
            path (``str`` | ``pathlib.Path``): The path of output file.
            obj (``dict`` | ``list`` | ``object``): The json data which need to be used for output. The type should be
                ``dict``, ``list`` or the ``object`` decorated by :func:`~tensorneko.io.text.text_reader.json_data`.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
            indent (``int``, optional): Python json dump indent parameter. Default: 4.
            ensure_ascii (``bool``, optional): Python json dump ensure_ascii parameter. Default: False.
            fast (``bool``, optional): Whether to use faster `orjson`. If `orjson` is not installed, use
                `json` library. Default: True
        """
        path = _path2str(path)
        if fast:
            try:
                import orjson
            except ImportError:
                fast = False
                warnings.warn("orjson is not installed, will use json lib instead.")
            assert encoding == "UTF-8", "orjson only supports UTF-8 encoding."
            assert indent in (0, 2, 4, None), "Only support indent 2, 4 or 0/None."
            assert ensure_ascii is False, "orjson does not support ensure_ascii."

        if type(obj) is dict:
            cls._write_json(path, obj, encoding, indent, ensure_ascii, fast)
        elif "is_json_data" in type(obj).__dict__ and type(obj).is_json_data:
            cls._write_json(path, obj.to_dict(), encoding, indent, ensure_ascii, fast)
        elif type(obj) is list:
            if len(obj) == 0 or type(obj[0]) in (dict, list):
                cls._write_json(path, obj, encoding, indent, ensure_ascii, fast)
            elif "is_json_data" in type(obj[0]).__dict__ and type(obj[0]).is_json_data:
                obj = [each.to_dict() for each in obj]
                cls._write_json(path, obj, encoding, indent, ensure_ascii, fast)
        else:
            raise TypeError("Not implemented type. Only support dict, list, json_data.")

    def __new__(cls, path: Union[str, Path], obj: Union[dict, list, object], encoding: str = "UTF-8", indent: int = 2,
        ensure_ascii: bool = False, fast: bool = True
    ) -> None:
        """Alias of :meth:`~tensorneko.io.json.json_writer.JsonWriter.to`."""
        path = _path2str(path)
        cls.to(path, obj, encoding, indent, ensure_ascii, fast)
