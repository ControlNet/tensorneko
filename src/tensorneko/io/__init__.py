from .reader import Reader
from .writer import Writer
from tensorneko_util.io import json_data, VisualLib

read = Reader()
write = Writer()

__all__ = [
    "read",
    "write",
    "json_data",
    "VisualLib"
]
