import os
from functools import reduce
from os.path import dirname, abspath
from typing import Callable, List, Dict, Iterable, Sequence, Any

from .fp import F, _, Stream
from .type import T, R


def identity(*args, **kwargs):
    """
    Pickle-friendly identity function.
    """
    assert kwargs == {}, "identity function doesn't accept keyword arguments"
    if len(args) == 1:
        return args[0]
    else:
        return args


def generate_inf_seq(items: Iterable[Any]) -> Stream:
    """
    Generate an infinity late-evaluate sequence.

    Args:
        items [``Iterable[Any]``]: Original items.

    Returns:
        :class:`~fp.stream.Stream`: The infinity sequence.

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


def with_printed(x: Any, func: Callable = identity) -> Any:
    """
    An identity function but with printed to console with some transform.

    Args:
        x (``Any``): Input.
        func (``Callable``, optional): A function used to apply the input for printing.

    Returns:
        ``Any``: Identity output.

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


def list_to_dict(l: List[T], key: Callable[[T], Any]) -> Dict[Any, T]:
    """
    Convert the list as a dictionary by given key.
    Args:
        l (``List[T]``): Input list.
        key (``(T) -> Any``): The key getter function.

    Returns:
        ``Dict[Any, T]``: The output dictionary.
    """
    return {key(x): x for x in l}


def get_tensorneko_util_path() -> str:
    """
    Get the `tensorneko_util` library root path

    Returns:
        ``str``: The root path of `tensorneko`
    """
    return dirname(dirname(abspath(__file__)))
