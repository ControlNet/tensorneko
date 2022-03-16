from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractServer(ABC):

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...

    def __enter__(self) -> AbstractServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
