from contextlib import contextmanager
from typing import TypeVar

from .bus import EventBus


class EventMeta(type):

    def __call__(cls, *args, **kwargs):
        event = super().__call__(*args, **kwargs)
        event.bus = kwargs.get("bus", EventBus.default)
        event.bus.emit(event, blocking=_blocking_flag)
        return event


_blocking_flag = True


@contextmanager
def no_blocking():
    """
    Context manager to disable blocking event handler.
    """
    global _blocking_flag
    _blocking_flag = False
    yield
    _blocking_flag = True


class Event(metaclass=EventMeta):
    bus: EventBus

    def __init_subclass__(cls, **kwargs):
        if "bus" in cls.__init__.__annotations__:
            raise TypeError("`bus` parameter is preserved. It should not be annotated in the __init__ method")


E = TypeVar("E", bound=Event)
