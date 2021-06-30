from abc import ABC, abstractmethod


class Configuration(ABC):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def __iter__(self):
        return iter((*self.args, *self.kwargs.values()))
