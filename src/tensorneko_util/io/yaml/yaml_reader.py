from typing import Type, Dict, Any, Union

import yaml
from yaml import Loader as Loader_, BaseLoader, FullLoader, SafeLoader, UnsafeLoader

L = Type[Union[Loader_, BaseLoader, FullLoader, SafeLoader, UnsafeLoader]]


class YamlReader:

    @classmethod
    def of(cls, path: str, Loader: L = Loader_) -> Dict[str, Any]:
        """
        Load a YAML file and return a Python dict.
        Args:
            path (``str``): YAML file path.
            Loader: (``Loader_ | BaseLoader | FullLoader | SafeLoader | UnsafeLoader``, optional): The loader for yaml. Default: :class:`yaml.Loader`

        Returns:
            ``dict``: The Python dict of given YAML file.
        """
        with open(path, "r") as file:
            return yaml.load(file, Loader=Loader)

    def __new__(cls, path: str, Loader: L = Loader_):
        """Alias of :meth:`~tensorneko_util.io.yaml.yaml_reader.YamlReader.of`."""
        return cls.of(path, Loader)
