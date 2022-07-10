from argparse import ArgumentParser
from typing import Any


class Arguments:
    def __init__(self):
        self._fields = {}

    def add(self, name, value):
        self._fields[name] = value
        self.__dict__[name] = value

    def __getitem__(self, item):
        return self._fields[item]

    def __str__(self):
        return f"Args {self._fields} "

    def __repr__(self):
        return self.__str__() if len(self._fields) < 10 else "Args (\n    " + \
            "\n    ".join([f"{k}={v}" for k, v in self._fields.items()]) + "\n)"


def get_parser_default_args(parser: ArgumentParser, **kwargs) -> Any:
    args = Arguments()
    for name, value in kwargs.items():
        args.add(name, value)
    for action in parser._actions:
        if action.dest not in ("help", "version") and action.dest not in args._fields:
            args.add(action.dest, action.default)
    return args
