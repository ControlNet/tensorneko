from typing import Type, Dict, Any, Union
from pathlib import Path
import yaml
from yaml import Loader as Loader_, BaseLoader, FullLoader, SafeLoader, UnsafeLoader

from .._path_conversion import _path2str

L = Type[Union[Loader_, BaseLoader, FullLoader, SafeLoader, UnsafeLoader]]


class YamlReader:

    @classmethod
    def of(cls, path: Union[str, Path], Loader: L = Loader_) -> Dict[str, Any]:
        """
        Load a YAML file and return a Python dict.
        Args:
            path (``str`` | ``pathlib.Path``): YAML file path.
            Loader: (``Loader_ | BaseLoader | FullLoader | SafeLoader | UnsafeLoader``, optional): The loader for yaml. Default: :class:`yaml.Loader`

        Returns:
            ``dict``: The Python dict of given YAML file.
        """
        path = _path2str(path)
        with open(path, "r") as file:
            return yaml.load(file, Loader=Loader)

    def __new__(cls, path: Union[str, Path], Loader: L = Loader_):
        """Alias of :meth:`~tensorneko_util.io.yaml.yaml_reader.YamlReader.of`."""
        path = _path2str(path)
        return cls.of(path, Loader)
