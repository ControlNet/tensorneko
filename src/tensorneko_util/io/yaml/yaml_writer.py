from typing import Any

import yaml


class YamlWriter:

    @classmethod
    def to(cls, path: str, obj: Any) -> None:
        """
        Write a Python object to a YAML file.

        Args:
            path (``str``): YAML file path.
            obj (``Any``): The Python object to write.
        """
        with open(path, "w") as f:
            yaml.dump(obj, f)

    def __new__(cls, path: str, obj: Any) -> None:
        """Alias of :meth:`~tensorneko_util.io.yaml.yaml_writer.YamlWriter.to`."""
        cls.to(path, obj)
        return None
