from .bus import EventBus


def subscribe(func):
    return EventBus.default.subscribe(func)


def subscribe_async(func):
    return EventBus.default.subscribe_async(func)


def subscribe_thread(func):
    return EventBus.default.subscribe_thread(func)


def subscribe_process(func):
    return EventBus.default.subscribe_process(func)
