from typing import Union
from pathlib import Path


def _path2str(path: Union[str, Path]) -> str:
    """
    Convert a path to a string.

    Args:
        path (``Union[str, Path]``): The path to be converted.

    Returns:
        ``str``: The string of the path.

    Examples::

        >>> _path2str("a/b/c")
        'a/b/c'
        >>> _path2str(Path("a/b/c"))
        'a/b/c'

    """
    if isinstance(path, Path):
        return str(path)
    return path
