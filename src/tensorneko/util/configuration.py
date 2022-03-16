from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Configuration(ABC):
    """
    Configuration base abstract class.

    This is designed for a functionality of using a configuration to build objects (layers, modules, etc.). Also, this
    configuration can also be saved to file and loaded from file.

    Examples:

        An example of configuration class for :class:`torch.nn.Linear`

        .. code-block:: python

            class LinearConfiguration(Configuration):

                def __init__(self, in_features, out_features, bias):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.bias = bias

                def build(self) -> torch.nn.Linear:
                    return torch.nn.Linear(self.in_features, self.out_features, self.bias)

                # using json to save and load conf
                def save(self, file_path: str) -> None:
                    attrs = {
                        "in_features": self.in_features,
                        "out_features": self.out_features,
                        "bias": self.bias
                    }
                    with open(file_path, "w", encoding="UTF-8") as file:
                        file.write(json.dumps(attrs))

                @staticmethod
                def load(file_path: str) -> LinearConfiguration:
                    with open(file_path, "r", encoding="UTF-8") as file:
                        attrs = json.loads(file.read())
                    return LinearConfiguration(attrs["in_features"], attrs["out_features"], attrs["bias"])


    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter((*self.args, *self.kwargs.values()))

    @abstractmethod
    def build(self) -> Any:
        """
        A method to build an object.

        Returns:
            ``Any``: The target object of this configuration.
        """
        ...

    @abstractmethod
    def save(self, *args, **kwargs):
        """Save this configuration to file."""
        ...

    @staticmethod
    @abstractmethod
    def load(*args, **kwargs) -> Configuration:
        """
        Load this configuration from file.

        Returns:
            :class:`Configuration`: This configuration object.
        """
        ...
