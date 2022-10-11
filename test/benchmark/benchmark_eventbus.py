from time import time

from tensorneko_util.util import Event, subscribe_thread
from tensorneko_util.util.eventbus import subscribe
from tensorneko_util.util.eventbus.event import no_blocking


# CPU-bound tasks

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


class FibEventSingleThread(Event):
    def __init__(self, n):
        self.n = n


for _ in range(10):
    @subscribe
    def fib_handler(event: FibEventSingleThread):
        fib(event.n)


class FibEventMultiThread(Event):
    def __init__(self, n):
        self.n = n


for _ in range(10):
    @subscribe_thread
    def fib_handler(event: FibEventMultiThread):
        fib(event.n)


if __name__ == '__main__':
    start = time()
    for _ in range(10):
        fib(30)
    print(f"Baseline: {time() - start} seconds")

    with no_blocking():

        start = time()
        FibEventSingleThread(30)
        print(f"EventBus Single-Thread: {time() - start} seconds")

        start = time()
        FibEventMultiThread(30)
        print(f"EventBus Multi-Thread: {time() - start} seconds")

