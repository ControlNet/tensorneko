from concurrent.futures import Future
from functools import wraps
from subprocess import Popen
from typing import Callable, Union, Any


def run_blocking(func: Callable[..., Union[Future, Popen]]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        task = func(*args, **kwargs)
        if isinstance(task, Future):
            return task.result()
        elif isinstance(task, Popen):
            task.wait()
            return task.returncode
        else:
            raise TypeError("run_blocking only works with Future or Popen")

    return wrapper
