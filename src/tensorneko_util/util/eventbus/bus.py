from __future__ import annotations

import asyncio
from concurrent.futures import wait
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, List, Callable, TYPE_CHECKING, Any, Coroutine, Union

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
    emit_type: EmitType


class EventBus:

    def __init__(self):
        self.listeners: Dict[Type[E], List[Listener]] = {}

    def register(self, event_type: Type[E], func: Callable[[E], None], emit_type: EmitType = EmitType.NORMAL) -> None:
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(Listener(func, emit_type))

    async def emit(self, event: E):
        event_type: Type[E] = type(event)
        if event_type in self.listeners:
            normal_listeners = []
            thread_listeners = []
            process_listeners = []
            async_listeners = []

            for listener in self.listeners[event_type]:
                if listener.emit_type == EmitType.NORMAL:
                    normal_listeners.append(listener)
                elif listener.emit_type == EmitType.THREAD:
                    thread_listeners.append(listener)
                elif listener.emit_type == EmitType.PROCESS:
                    process_listeners.append(listener)
                elif listener.emit_type == EmitType.ASYNC:
                    async_listeners.append(listener)
                else:
                    raise ValueError(f"Unknown emit type: {listener.emit_type}")

            # priority: Process > Thread > Async > Normal
            futures = []
            for listener in process_listeners:
                futures.append(ExecutorPool.submit(listener.func, event, parallel_type=ParallelType.PROCESS))
            for listener in thread_listeners:
                futures.append(ExecutorPool.submit(listener.func, event, parallel_type=ParallelType.THREAD))
            coroutines = []
            for listener in async_listeners:
                coroutines.append(asyncio.create_task(listener.func(event)))
            for listener in normal_listeners:
                listener.func(event)

            for coroutine in coroutines:
                await coroutine
            wait(futures)


    def subscribe(self,
        func: Callable[[E], Union[None, Coroutine[Any, Any, None]]],
        emit_type: EmitType = EmitType.NORMAL
    ) -> Callable[[E], None]:
        if "event" not in func.__annotations__:
            raise TypeError("@subscribe decorator must follow the signature `handler_func(event: EventType)`. "
                            "The event type should be annotated.")

        event_type = func.__annotations__["event"]
        self.register(event_type, func, emit_type)
        return func

    def subscribe_async(self, func: Callable[[E], Coroutine[Any, Any, None]]):
        return self.subscribe(func, emit_type=EmitType.ASYNC)

    def subscribe_thread(self, func: Callable[[E], None]):
        return self.subscribe(func, emit_type=EmitType.THREAD)

    def subscribe_process(self, func: Callable[[E], None]):
        return self.subscribe(func, emit_type=EmitType.PROCESS)


bus = EventBus()
