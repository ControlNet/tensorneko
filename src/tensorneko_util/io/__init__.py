from .reader import Reader
from .writer import Writer

from .json import json_data

read = Reader()
writer = Writer()

__all__ = [
    "read",
    "writer",
    "json_data"
]
