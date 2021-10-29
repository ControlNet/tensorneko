from __future__ import annotations

import warnings
from typing import Callable, Dict, List

import inspect


class Dispatcher:
    dispatchers: Dict[str, Dispatcher] = {}

    def __init__(self, name: str):
        self.name = name
        self._functions = {}

    def __call__(self, func) -> Callable:
        if isinstance(func, (classmethod, staticmethod)):
            func = func.__func__

        parameters: List[inspect.Parameter] = [*inspect.signature(func).parameters.values()]
        possible_types = [[]]

        for param in parameters:
            # if is union type
            is_union = str(param.annotation)[:13] == "typing.Union["

            # if no default value and not union, just append to each possible types
            if param.default is inspect.Signature.empty and not is_union:
                for types in possible_types:
                    types.append(param.annotation)
            # if no default value but is union, append to each possible types and add new types
            elif param.default is inspect.Signature.empty and is_union:
                new_possible_types = []
                # add new types
                for i in range(len(possible_types)):
                    types = possible_types[i]
                    for type_ in param.annotation.__args__[1:]:
                        new_possible_types.append(types + [type_])
                # append first union arg to each possible types
                for types in possible_types:
                    types.append(param.annotation.__args__[0])
                possible_types.extend(new_possible_types)
            # if has default value and is not union
            elif param.default is not inspect.Signature.empty and not is_union:
                for i in range(len(possible_types)):
                    types = possible_types[i]
                    possible_types.append(types + [param.annotation])
            # if has default value and is union
            elif param.default is not inspect.Signature.empty and is_union:
                for i in range(len(possible_types)):
                    types = possible_types[i]
                    for type_ in param.annotation.__args__:
                        possible_types.append(types + [type_])
            else:
                raise TypeError(f"Unknown Error in Dispatcher with {parameters}")

        for types in possible_types:
            if tuple(types) in self._functions:
                warnings.warn(f"The dispatcher in {func.__name__}({str(types)[1:-1]}) is overridden!", stacklevel=2)
            self._functions[tuple(types)] = func

        return Resolver(self)

    def __class_getitem__(cls, key) -> Dispatcher:
        if key in cls.dispatchers:
            return cls.dispatchers[key]
        else:
            new_obj = Dispatcher(key)
            cls.dispatchers[key] = new_obj
            return new_obj


def dispatch(func):
    name = ".".join([func.__module__, func.__qualname__])
    return Dispatcher[name](func)


class Resolver:

    def __init__(self, dispatcher: Dispatcher):
        self._dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args] + [type(kwarg) for kwarg in kwargs.values()])
        if types in self._dispatcher._functions:
            return self._dispatcher._functions[types](*args, **kwargs)
        else:
            raise TypeError(f"Not valid for type {str(types)} for function {self._dispatcher.name}")
