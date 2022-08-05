import json
from typing import Union, Optional

from ...util.type import T


class JsonReader:

    @classmethod
    def of(cls, path: str, clazz: Optional[T] = None, encoding: str = "UTF-8") -> Union[T, dict, list]:
        """
        Read json files to ``list`` or ``dict``.

        Args:
            path (``str``): Json file path.
            clazz: (``T``, optional): The object of the json read for. The type should be decorated by
                :func:`json_data`. This should be ``T`` or ``List[T]`` or ``List[List[T]]``
            encoding (``str``, optional): The encoding for python ``open`` function. Default: "UTF-8"

        Returns:
            ``dict`` | ``list`` | ``object``: The object of given json.
        """
        if clazz is not None:
            if "typing.List[" in str(clazz):
                with open(path, "r", encoding=encoding) as file:
                    obj: list = json.loads(file.read())

                inner_type = clazz.__args__[0]
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
                    obj = clazz(json.loads(file.read()))
        else:
            with open(path, "r", encoding=encoding) as file:
                obj = json.loads(file.read())

        return obj

    def __new__(cls, path: str, clazz: T = None, encoding: str = "UTF-8") -> Union[T, dict, list]:
        """Alias of :meth:`~tensorneko_util.io.json.json_reader.JsonReader.of`."""
        return cls.of(path, clazz, encoding)
