from typing import Any, Union
from pathlib import Path
import yaml

from .._path_conversion import _path2str


class YamlWriter:

    @classmethod
    def to(cls, path: Union[str, Path], obj: Any) -> None:
        """
        Write a Python object to a YAML file.

        Args:
            path (``str`` | ``pathlib.Path``): YAML file path.
            obj (``Any``): The Python object to write.
        """
        path = _path2str(path)
        with open(path, "w") as f:
            yaml.dump(obj, f)

    def __new__(cls, path: Union[str, Path], obj: Any) -> None:
        """Alias of :meth:`~tensorneko_util.io.yaml.yaml_writer.YamlWriter.to`."""
        path = _path2str(path)
        cls.to(path, obj)
