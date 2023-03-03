import json
from typing import Union


class JsonWriter:

    @classmethod
    def to(cls, path: str, obj: Union[dict, list, object], encoding: str = "UTF-8", indent=4, ensure_ascii=False
    ) -> None:
        """
        Save as Json file from a dictionary, list or json_dict.

        Args:
            path (``str``): The path of output file.
            obj (``dict`` | ``list`` | ``object``): The json data which need to be used for output. The type should be
                ``dict``, ``list`` or the ``object`` decorated by :func:`~tensorneko.io.text.text_reader.json_data`.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
            indent (``int``, optional): Python json dump indent parameter. Default: 4.
            ensure_ascii (``bool``, optional): Python json dump ensure_ascii parameter. Default: False.
        """
        if type(obj) is dict:
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))
        elif "is_json_data" in type(obj).__dict__ and type(obj).is_json_data:
            with open(path, "w", encoding=encoding) as file:
                file.write(json.dumps(obj.to_dict(), indent=indent, ensure_ascii=ensure_ascii))
        elif type(obj) is list:
            if len(obj) == 0 or type(obj[0]) in (dict, list):
                with open(path, "w", encoding=encoding) as file:
                    file.write(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))
            elif "is_json_data" in type(obj[0]).__dict__ and type(obj[0]).is_json_data:
                obj = [each.to_dict() for each in obj]
                with open(path, "w", encoding=encoding) as file:
                    file.write(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))
        else:
            raise TypeError("Not implemented type. Only support dict, list, json_data.")

    def __new__(cls, path: str, obj: Union[dict, list, object], encoding: str = "UTF-8", indent=4, ensure_ascii=False
    ) -> None:
        """Alias of :meth:`~tensorneko.io.json.json_writer.JsonWriter.to`."""
        cls.to(path, obj, encoding, indent, ensure_ascii)
