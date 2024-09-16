import json
import warnings
from typing import Union, Optional, List, Union
from pathlib import Path

from .._path_conversion import _path2str
from ...util.type import T


class JsonReader:

    @staticmethod
    def _read_json(path: str, encoding: str = "UTF-8", fast: bool = False) -> Union[dict, list]:
        if not fast:
            with open(path, "r", encoding=encoding) as file:
                return json.load(file)
        else:
            import orjson
            with open(path, "rb") as file:
                return orjson.loads(file.read())

    @classmethod
    def of(cls, path: Union[str, Path], clazz: Optional[T] = None, encoding: str = "UTF-8", fast: bool = True
    ) -> Union[T, dict, list]:
        """
        Read json files to ``list`` or ``dict``.

        Args:
            path (``str`` | ``pathlib.Path``): Json file path.
            clazz: (``T``, optional): The object of the json read for. The type should be decorated by
                :func:`json_data`. This should be ``T`` or ``List[T]`` or ``List[List[T]]``
            encoding (``str``, optional): The encoding for python ``open`` function. Default: "UTF-8"
            fast (``bool``, optional): Whether to use faster `orjson`. If `orjson` is not installed, use
                `json` library. Default: True

        Returns:
            ``dict`` | ``list`` | ``object``: The object of given json.
        """
        path = _path2str(path)
        if fast:
            try:
                import orjson
            except ImportError:
                fast = False
                warnings.warn("orjson is not installed, will use json lib instead.")
            assert encoding == "UTF-8", "orjson only supports UTF-8 encoding."

        if clazz is not None:
            if "typing.List[" in str(clazz):
                obj = cls._read_json(path, encoding, fast)

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
                obj = clazz(cls._read_json(path, encoding, fast))
        else:
            obj = cls._read_json(path, encoding, fast)

        return obj


    @classmethod
    def of_jsonl(cls, path: Union[str, Path], clazz: Optional[T] = None, encoding: str = "UTF-8", fast: bool = True
    ) -> List[Union[T, dict, list]]:
        """
        Read jsonl files to ``list`` or ``dict``.

        Args:
            path (``str`` | ``pathlib.Path``): Jsonl file path.
            clazz: (``T``, optional): The object of the jsonl read for. The type should be decorated by
                :func:`json_data`. This should be ``T`` or ``List[T]`` or ``List[List[T]]``
            encoding (``str``, optional): The encoding for python ``open`` function. Default: "UTF-8"
            fast (``bool``, optional): Whether to use faster `orjson`. If `orjson` is not installed, use
                `json` library. Default: True

        Returns:
            ``List[dict]`` | ``List[list]`` | ``List[T]``: The object of given jsonl.
        """
        path = _path2str(path)
        if fast:
            try:
                import orjson
            except ImportError:
                fast = False
                warnings.warn("orjson is not installed, will use json lib instead.")
            assert encoding == "UTF-8", "orjson only supports UTF-8 encoding."

        if clazz is not None:
            if "typing.List[" in str(clazz):
                with open(path, "r", encoding=encoding) as file:
                    if fast:
                        obj: list = [orjson.loads(line) for line in file]
                    else:
                        obj: list = [json.loads(line) for line in file]

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
                    if fast:
                        obj: list = [orjson.loads(line) for line in file]
                    else:
                        obj: list = [json.loads(line) for line in file]
                    obj = clazz(obj)
        else:
            with open(path, "r", encoding=encoding) as file:
                if fast:
                    obj: list = [orjson.loads(line) for line in file]
                else:
                    obj: list = [json.loads(line) for line in file]

        return obj

    def __new__(cls, path: Union[str, Path], clazz: T = None, encoding: str = "UTF-8", fast: bool = True) -> Union[T, dict, list]:
        """
        Read json or jsonl file smartly.

        Args:
            path (``str`` | ``pathlib.Path``): Json or jsonl file path.
            clazz: (``T``, optional): The object of the json read for. The type should be decorated by
                :func:`json_data`. This should be ``T`` or ``List[T]`` or ``List[List[T]]``
            encoding (``str``, optional): The encoding for python ``open`` function. Default: "UTF-8"
            fast (``bool``, optional): Whether to use faster `orjson`. If `orjson` is not installed, use
                `json` library. Default: True

        Returns:
            ``dict`` | ``list`` | ``object``: The object of given json or jsonl.
        """
        path = _path2str(path)
        if path.endswith(".jsonl"):
            return cls.of_jsonl(path, clazz, encoding, fast)
        else:
            return cls.of(path, clazz, encoding, fast)
