from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import wait
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, Callable, TYPE_CHECKING, Any, Coroutine, Union, Set, TypeVar

from ...backend.parallel import ExecutorPool, ParallelType

if TYPE_CHECKING:
    from .event import E


class EmitType(Enum):
    NORMAL = 0
    THREAD = 1
    PROCESS = 2
    ASYNC = 3


@dataclass
class Listener:
    func: Callable[[E], Union[None, Coroutine[Any, Any, None]]]
    func_name: str
    emit_type: EmitType

    func_names = set()

    def __post_init__(self):
        if self.func.__name__ in self.func_names and self.emit_type == EmitType.PROCESS:
            warnings.warn(f"Multiprocess event handler {self.func_name} is already registered")
        self.func_names.add(self.func_name)

    def __hash__(self):
        return hash(self.func) + hash(self.emit_type)


class EventHandler(ABC):
    """
    EventHandler is an abstract class that is used to define stateful event handlers.

    The `__init__` method should not have any arguments.

    The `__call__` method should include only one argument `event: EventType` and will be invoked when an event is
        emitted.

    Examples::

        from tensorneko.util import Event, EventHandler, subscribe

        class AddValueEvent(Event):
            def __init__(self, value: int):
                self.value = value

        @subscribe
        class CustomHandler(EventHandler):
            def __init__(self):
                self.x = 0

            def __call__(self, event: AddValueEvent):
                self.x += event.value

        if __name__ == '__main__':
            AddValueEvent(42)
            assert CustomHandler.instance().x == 42

    """

    _instance = None

    @abstractmethod
    def __call__(self, event: E):
        pass

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


H = TypeVar("H", bound=EventHandler)


class EventBus:
    """
    Event bus can be used to program the reactive behavior of the system. A default bus is provided as `bus`, and it
    can be imported as `from tensorneko.util import bus`.

    The stateless event handler can be registered to the default bus by using the `@subscribe` decorator.

    Examples:

        Use default event bus.

        .. code-block:: python

            # useful decorators for default event bus
            from tensorneko.util import (
                subscribe, # run in the main thread
                subscribe_thread, # run in a new thread
                subscribe_async, # run async
                subscribe_process # run in a new process
            )
            # Event base type
            from tensorneko.util import Event

            class LogEvent(Event):
                def __init__(self, message: str):
                    self.message = message

            # the event argument should be annotated correctly
            @subscribe
            def log_information(event: LogEvent):
                print(event.message)

            @subscribe_thread
            def log_information_thread(event: LogEvent):
                print(event.message, "in another thread")

            @subscribe_async
            async def log_information_async(event: LogEvent):
                print(event.message, "async")

            @subscribe_process
            def log_information_process(event: LogEvent):
                print(event.message, "in a new process")

            if __name__ == '__main__':
                # emit an event, and then the event handler will be invoked
                # The sequential order is not guaranteed
                LogEvent("Hello world!")

                # one possible output:
                # Hello world! in another thread
                # Hello world! async
                # Hello world!
                # Hello world! in a new process

        Create multiple event bus

        .. code-block:: python

            from tensorneko.util import EventBus
            bus1 = EventBus()
            bus2 = EventBus()

            # register event handlers to different event bus
            @bus1.subscribe
            def log_information(event: LogEvent):
                print(event.message, "in bus1")

            @bus2.subscribe
            def log_information(event: LogEvent):
                print(event.message, "in bus2")

            if __name__ == '__main__':
                # emit the event with specified bus
                LogEvent("Hello world!", bus=bus1)
                LogEvent("Hello world!", bus=bus2)

                # one possible output:
                # Hello world! in bus1
                # Hello world! in bus2
    """

    default: EventBus = None

    def __init__(self):
        self.listeners: Dict[Type[E], Set[Listener]] = {}

    def register(self,
        event_type: Type[E],
        func: Callable[[E], None],
        func_name: str = None,
        emit_type: EmitType = EmitType.NORMAL
    ) -> None:
        if func_name is None:
            func_name = func.__name__

        if event_type not in self.listeners:
            self.listeners[event_type] = set()
        self.listeners[event_type].add(Listener(func, func_name, emit_type))

    def emit(self, event: E, blocking: bool = True) -> None:
        event_type: Type[E] = type(event)
        if event_type in self.listeners:
            normal_listeners = set()
            thread_listeners = set()
            process_listeners = set()
            async_listeners = set()

            for listener in self.listeners[event_type]:
                if listener.emit_type == EmitType.NORMAL:
                    normal_listeners.add(listener)
                elif listener.emit_type == EmitType.THREAD:
                    thread_listeners.add(listener)
                elif listener.emit_type == EmitType.PROCESS:
                    process_listeners.add(listener)
                elif listener.emit_type == EmitType.ASYNC:
                    async_listeners.add(listener)
                else:
                    raise ValueError(f"Unknown emit type: {listener.emit_type}")

            # priority: Process > Thread > Async > Normal
            futures = []
            for listener in process_listeners:
                futures.append(ExecutorPool.submit(listener.func, event, parallel_type=ParallelType.PROCESS))
            for listener in thread_listeners:
                futures.append(ExecutorPool.submit(listener.func, event, parallel_type=ParallelType.THREAD))
            if len(async_listeners) > 0:
                asyncio.run(self.emit_async(async_listeners, event))
            for listener in normal_listeners:
                listener.func(event)

            if blocking:
                wait(futures)

    @staticmethod
    async def emit_async(async_listeners: Set[Listener], event: E) -> None:
        coroutines = []
        for listener in async_listeners:
            coroutines.append(asyncio.create_task(listener.func(event)))
        for coroutine in coroutines:
            await coroutine

    def subscribe(self,
        func: Union[Callable[[E], Union[None, Coroutine[Any, Any, None]]], Type[H]],
        emit_type: EmitType = EmitType.NORMAL
    ) -> Callable[[E], None]:
        if isinstance(func, type):
            event_cls = func
            if not issubclass(event_cls, EventHandler):
                raise ValueError(f"{event_cls.__name__} is not inherited from EventHandler")

            event_handler = event_cls.instance()
            func = event_handler.__call__
            func_name = event_handler.__class__.__name__
            return_value = event_cls
        else:
            func_name = func.__name__
            return_value = func

        if "event" not in func.__annotations__:
            raise TypeError("@subscribe decorator must follow the signature `handler_func(event: EventType)`. "
                            "The event type should be annotated.")

        event_type = func.__annotations__["event"]
        self.register(event_type, func, func_name, emit_type)

        return return_value

    def subscribe_async(self, func: Callable[[E], Coroutine[Any, Any, None]]):
        return self.subscribe(func, emit_type=EmitType.ASYNC)

    def subscribe_thread(self, func: Callable[[E], None]):
        return self.subscribe(func, emit_type=EmitType.THREAD)

    def subscribe_process(self, func: Callable[[E], None]):
        return self.subscribe(func, emit_type=EmitType.PROCESS)


EventBus.default = EventBus()
