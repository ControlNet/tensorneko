from typing import TypeVar

from .bus import EventBus


class EventMeta(type):

    def __call__(cls, *args, **kwargs):
        event = super().__call__(*args, **kwargs)
        event.bus = kwargs.get("bus", EventBus.default)
        event.bus.emit(event)
        return event


class Event(metaclass=EventMeta):
    bus: EventBus

    def __init_subclass__(cls, **kwargs):
        if "bus" in cls.__init__.__annotations__:
            raise TypeError("`bus` parameter is preserved. It should not be annotated in the __init__ method")


E = TypeVar("E", bound=Event)
