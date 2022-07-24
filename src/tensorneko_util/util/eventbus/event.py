from typing import TypeVar

from .bus import EventBus


class EventMeta(type):

    def __call__(cls, *args, bus=EventBus.default, **kwargs):
        event = super().__call__(*args, **kwargs)
        bus.emit(event)
        return event


class Event(metaclass=EventMeta):
    bus: EventBus


E = TypeVar("E", bound=Event)
