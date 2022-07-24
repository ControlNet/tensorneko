import asyncio
from typing import TypeVar

from .bus import bus as default_bus, EventBus


class EventMeta(type):

    def __call__(cls, *args, bus=default_bus, **kwargs):
        event = super().__call__(*args, **kwargs)
        asyncio.run(bus.emit(event))
        return event


class Event(metaclass=EventMeta):
    bus: EventBus


E = TypeVar("E", bound=Event)
