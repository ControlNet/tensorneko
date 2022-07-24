from .bus import bus, EmitType


def subscribe(func):
    return bus.subscribe(func)


def subscribe_async(func):
    return bus.subscribe(func, emit_type=EmitType.ASYNC)


def subscribe_thread(func):
    return bus.subscribe(func, emit_type=EmitType.THREAD)


def subscribe_process(func):
    return bus.subscribe(func, emit_type=EmitType.PROCESS)
