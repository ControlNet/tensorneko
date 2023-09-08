from typing import Callable, Coroutine, Any

from .bus import EventBus
from .event import E


class SubscribeDecorator:

    def __call__(self, func: Callable[[E], None]):
        return EventBus.default.subscribe(func)

    def coro(self, func: Callable[[E], Coroutine[Any, Any, None]]):
        return EventBus.default.subscribe_async(func)

    def thread(self, func: Callable[[E], None]):
        return EventBus.default.subscribe_thread(func)

    def process(self, func: Callable[[E], None]):
        return EventBus.default.subscribe_process(func)


subscribe = SubscribeDecorator()
