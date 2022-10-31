import time
from functools import wraps
from typing import Callable, Optional


class Timer:
    """
    A timer to count the time of a function or a block of code.

    Examples::

        from tensorneko.util import Timer
        import time

        # use as a context manager with single time
        with Timer():
            time.sleep(1)

        # use as a context manager with multiple segments
        with Timer() as t:
            time.sleep(1)
            t.time("sleep A")
            time.sleep(1)
            t.time("sleep B")
            time.sleep(1)

        # use as a decorator
        @Timer()
        def f():
            time.sleep(1)
            print("f")

    """

    def __init__(self):
        self.times = []
        self.names = []
        self.total_time = None

    def __enter__(self):
        self.times.append(time.perf_counter())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_time = time.perf_counter() - self.times[0]
        print(self._make_str("Total", self.total_time))

    def time(self, name: Optional[str] = None):
        self.times.append(time.perf_counter())
        name = name or ""
        print(self._make_str(name, time.perf_counter() - self.times[-2]))

    def __call__(self, f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            self.times.append(time.perf_counter())
            result = f(*args, **kwargs)
            print(self._make_str(f.__name__, time.perf_counter() - self.times[-1]))
            return result

        return wrapper

    @staticmethod
    def _make_str(name: str, t: float):
        return f"[Timer] {name}: {t} sec"

