import importlib.util
import os
import time
from functools import reduce
from os.path import dirname, abspath
from types import ModuleType
from typing import Callable, List, Dict, Iterable, Sequence, Any, Optional, Type, Union, Tuple

import numpy as np

from .fp import F, _, Stream
from .type import T, R, T_E


def identity(*args, **kwargs):
    """
    Pickle-friendly identity function.
    """
    assert kwargs == {}, "identity function doesn't accept keyword arguments"
    if len(args) == 1:
        return args[0]
    else:
        return args


def generate_inf_seq(items: Iterable[T]) -> Stream[T]:
    """
    Generate an infinity late-evaluate sequence.

    Args:
        items [``Iterable[T]``]: Original items.

    Returns:
        :class:`~fp.stream.Stream[T]`: The infinity sequence.

    Examples::

        >>> seq = generate_inf_seq(["a", 1, ["e0", "e1"]])
        >>> list(seq[:10])
        ['a', 1, ['e0', 'e1'], 'a', 1, ['e0', 'e1'], 'a', 1, ['e0', 'e1'], 'a']

    Note:
        The repeated reference type objects are the same object.

        >>> a_list = ["a", "b"]
        >>> seq = list(generate_inf_seq([a_list])[:3])
        >>> seq
        [['a', 'b'], ['a', 'b'], ['a', 'b']]
        >>> a_list.append("c")
        >>> seq
        [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']]

    """
    s = Stream()
    return s << items << map(_, s)


def compose(fs: Sequence[Callable]) -> F:
    """
    Compose functions as a pipeline function.

    Args:
        fs (``Sequence[Callable]``): The functions input for composition.

    Returns:
        :class:`~fp.func.F`: The composed output function.

    Examples::

        >>> f = lambda x: x + 1
        >>> g = lambda x: x * 2
        >>> h = lambda x: x ** 2
        >>> x = 1
        >>> h(g(f(x))) == compose([f, g, h])(x)
        True

    """
    return reduce(_ >> _, fs, F())


def listdir(path: str, filter_func: Callable[[str], bool] = lambda arg: True) -> List[str]:
    """
    The full path version of :func:`os.listdir`.

    Get the all listdir result with input path.

    Args:
        path (``str``): The path for listdir.
        filter_func (``(str) -> bool``, optional): The filter function to filter the file/directory name in ``listdir``.

    Returns:
        ``List[str]``: listdir result with input path.

    Examples::

        >>> os.listdir("tensorneko/util")[:3]  # python os library
        ['configuration.py', 'enum.py', 'string_getter.py']
        >>> listdir("tensorneko/util")[:3]  # tensorneko
        ['tensorneko/util/configuration.py',
         'tensorneko/util/enum.py',
         'tensorneko/util/string_getter.py']

    """
    files = filter(filter_func, os.listdir(path))
    return list(map(F(os.path.join, path), files))


def with_printed(x: T, func: Callable[[T], Any] = identity) -> T:
    """
    An identity function but with printed to console with some transform.

    Args:
        x (``T``): Input.
        func (``(T) -> Any``, optional): A function used to apply the input for printing. Default: identity function.

    Returns:
        ``T``: Identity output.

    Examples::

        >>> x = torch.tensor([1.5, 2.5, 3.5])
        >>> y = with_printed(x, lambda tensor: tensor.dtype)
        torch.float32
        >>> x == y
        tensor([True, True, True])

    """
    print(func(x))
    return x


def ifelse(predicate: Callable[[T], bool], func_true: Callable[[T], R], func_false: Callable[[T], R] = identity
) -> Callable[[T], R]:
    """
    A function composition util for if-else control flow.

    Args:
        predicate (``(T) -> bool``): A predicate produce a bool.
        func_true (``(T) -> R``): If the bool from predicate is True, return this function.
        func_false (``(T) -> R``): If the bool from predicate is False, return this function.

    Returns:
        ``(T) -> R``: Composed function with if-else control flow.
    """

    def wrapper(*args, **kwargs):
        if predicate(*args, **kwargs):
            return func_true(*args, **kwargs)
        else:
            return func_false(*args, **kwargs)

    return wrapper


def dict_add(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries.

    Args:
        *dicts: dictionaries to merge.

    Returns:
        ``dict``: The merged dictionary.
    """
    new_dict = {}
    for each in dicts:
        new_dict.update(each)
    return new_dict


def as_list(*args, **kwargs) -> list:
    """
    Returns:
        ``list``: The list of args.

    Examples::

        >>> as_list(1, 2, 3, 4, key=5)
        [1, 2, 3, 4, 5]
    """
    return list(args) + list(kwargs.values())


def list_to_dict(l: List[T], key: Callable[[T], R]) -> Dict[R, T]:
    """
    Convert the list as a dictionary by given key.
    Args:
        l (``List[T]``): Input list.
        key (``(T) -> R``): The key getter function.

    Returns:
        ``Dict[R, T]``: The output dictionary.
    """
    return {key(x): x for x in l}


def get_tensorneko_util_path() -> str:
    """
    Get the `tensorneko_util` library root path

    Returns:
        ``str``: The root path of `tensorneko`
    """
    return dirname(dirname(abspath(__file__)))


def circular_pad(x: List[T], target: int) -> List[T]:
    """
    Circular padding a list to the target length.

    Args:
        x(``List[T]``): Input list.
        target(``int``): Target length.

    Returns:
        ``List[T]``: The padded list.
    """
    if len(x) == target:
        return x
    elif len(x) > target:
        return x[:target]
    elif 2 * len(x) > target:
        return x + x[:target - len(x)]
    else:
        return circular_pad(x + x, target)


def load_py(path: str) -> ModuleType:
    """
    Load a python file as a module.

    Args:
        path (``str``): The path of the python file.

    Returns:
        ``Any``: The loaded module.
    """
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def try_until_success(func: Callable[..., T], *args, max_trials: Optional[int] = None, sleep_time: int = 0,
    exception_callback: Optional[Callable[[Exception], None]] = None,
    exception_type: Union[Type[T_E], Tuple[T_E, ...]] = Exception, **kwargs
) -> T:
    """
    Try to run the function until success.

    Args:
        func (``(...) -> T``): The function to run.
        max_trials (``int``, optional): The max try times. None for unlimited. Default: None.
        sleep_time (``int``, optional): The sleep time between each try. Default: 0.
        exception_callback (``(Exception) -> None``, optional): The callback function when exception occurs.
        exception_type (``Type[Exception] | (Type[Exception], ...)``, optional): The exception types to catch.
            Default: Exception.
        *args: The args for the function.
        **kwargs: The kwargs for the function.

    Returns:
        ``T``: The output of the function.

    Raises:
        ``Exception``: The exception from the function.
    """
    trials = 0
    while True:
        try:
            return func(*args, **kwargs)
        except exception_type as e:
            trials += 1
            if exception_callback is not None:
                exception_callback(e)
            if max_trials is not None and trials > max_trials:
                raise e
        if sleep_time > 0:
            time.sleep(sleep_time)


def sample_indexes(total_frames: int, n_frames: int, sample_rate: int) -> np.ndarray:
    """
    Sample continuous indexes from total frames.

    Args:
        total_frames (``int``): The total number of items from source.
        n_frames (``int``): The number of items to sample.
        sample_rate (``int``): The sample rate to retrieve the continuous items.
    Returns:
        ``np.ndarray``: The sampled indexes.

    Examples::

        >>> sample_indexes(10, 3, 1)
        array([5, 6, 7])

        >>> sample_indexes(10, 3, 2)
        array([1, 3, 5])

        >>> sample_indexes(10, 3, 3)
        array([2, 5, 8])

    """

    start_ind = np.random.randint(0, total_frames - (n_frames * sample_rate) + 2, ())
    return np.arange(n_frames) * sample_rate + start_ind
