from __future__ import annotations

import time
import inspect
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

        # without parentheses
        @Timer
        def g():
            time.sleep(1)
            print("f")

        # disable verbose and get time manually
        with Timer(verbose=False) as t:
            time.sleep(1)
            dt = t.time("sleep A")
            print(f"Time of sleep A: {dt}")
            time.sleep(1)
            dt = t.time("sleep B")
            print(f"Time of sleep B: {dt}")
            print(f"Time of sleep A and B: {t.elapsed}")
            time.sleep(1)
            print(f"Total time: {t.elapsed}")

    """

    def __init__(self, verbose: bool = True):
        self.times = []
        self.names = []
        self.total_time = None
        self.verbose = verbose

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Callable):
            return cls()(args[0])
        else:
            return super().__new__(cls)

    def __enter__(self):
        self.times.append(time.perf_counter())
        return self

    @property
    def elapsed(self):
        return time.perf_counter() - self.times[0]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_time = self.elapsed
        if self.verbose:
            print(self._make_str("Total", self.total_time))

    def time(self, name: Optional[str] = None) -> float:
        self.times.append(time.perf_counter())
        name = name or ""
        dt = time.perf_counter() - self.times[-2]
        if self.verbose:
            print(self._make_str(name, dt))
        return dt

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

    @classmethod
    def lines(cls, __f: Callable) -> Callable:
        source_code = inspect.getsource(__f)
        # first line should be started from `def`
        source_code_lines = source_code.split("\n")
        new_source_code_lines = []
        # get base indent for the function body
        base_indent = None
        body_start_index = 0
        for i, line in enumerate(source_code_lines):
            if line == "" or line.strip() == "@Timer.lines":
                continue
            if line.lstrip().startswith("def") or line.lstrip().startswith("@"):
                new_source_code_lines.append(line)
                body_start_index = i
            else:
                if base_indent is None:
                    # initialize the timer
                    base_indent = line.index(line.lstrip())
                    new_source_code_lines.append(f"{' ' * base_indent}__timer = Timer().__enter__()")

                # skip the if, for, this kind of lines ended with :
                if line.rstrip().endswith(":"):
                    new_source_code_lines.append(line)
                # exit the timer before return
                elif line.lstrip().startswith("return"):
                    return_line_indent = line.index(line.lstrip())
                    new_source_code_lines.append(f"{' ' * base_indent}__timer.__exit__(None, None, None)")
                    new_source_code_lines.append(line)
                else:
                    new_source_code_lines.append(line + f'; __timer.time(name="{__f.__name__} line {i - body_start_index}")')

        new_source_code = "\n".join(new_source_code_lines) + "\n"
        exec(new_source_code)
        return locals()[__f.__name__]
