from contextlib import contextmanager
from typing import TypeVar

from .bus import EventBus


class EventMeta(type):

    def __call__(cls, *args, bus=EventBus.default, **kwargs):
        event = super().__call__(*args, **kwargs)
        bus.emit(event, blocking=_blocking_flag)
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


E = TypeVar("E", bound=Event)
