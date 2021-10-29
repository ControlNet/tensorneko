from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, Dict, List, Generic, Sequence

import inspect

from .type import T


class DispatcherTypeWarning(Warning):
    pass


class Dispatcher:
    dispatchers: Dict[str, Dispatcher] = {}

    def __init__(self, name: str):
        self.name = name
        self._functions = {}

    def __call__(self, func: Callable[..., T], set_types: Sequence[type] = None) -> Resolver[T]:
        if isinstance(func, (classmethod, staticmethod)):
            func = func.__func__
        if not set_types:
            parameters: List[inspect.Parameter] = [*inspect.signature(func).parameters.values()]

            if parameters[0].name == "cls" and parameters[0].annotation is inspect.Parameter.empty:
                # if it is a class method
                parameters[0] = inspect.Parameter("cls", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=type)

            is_method = False
            if parameters[0].name == "self" and parameters[0].annotation is inspect.Parameter.empty:
                # if the function is a method
                parameters = parameters[1:]
                is_method = True

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

            has_empty_type = False
            for types in possible_types:
                if tuple(types) in self._functions:
                    warnings.warn(f"The dispatcher in {func.__module__ + '.' + func.__name__}({str(types)[1:-1]}) "
                                  f"is overridden!",
                        DispatcherTypeWarning, stacklevel=2)

                if inspect.Signature.empty in types:
                    warnings.warn(f"The dispatcher in {func.__module__ + '.' + func.__name__}({str(types)[1:-1]}) "
                                  f"has no type annotation!",
                        DispatcherTypeWarning, stacklevel=2)
                    has_empty_type = True
                self._functions[tuple(types)] = func

            if not is_method:
                return Resolver(self)
            else:
                return MethodResolver(self)
        else:
            self._functions[tuple(set_types)] = func
            return Resolver(self)

    @classmethod
    def get(cls, key: str) -> Dispatcher:
        if key in cls.dispatchers:
            return cls.dispatchers[key]
        else:
            new_obj = Dispatcher(key)
            cls.dispatchers[key] = new_obj
            return new_obj


class Resolver(Generic[T]):

    def __init__(self, dispatcher: Dispatcher):
        self._dispatcher = dispatcher

    def __call__(self, *args, **kwargs) -> T:
        types = tuple([type(arg) for arg in args] + [type(kwarg) for kwarg in kwargs.values()])
        if types in self._dispatcher._functions:
            return self._dispatcher._functions[types](*args, **kwargs)
        else:
            raise TypeError(f"Not valid for type {str(types)} for function {self._dispatcher.name}")


class MethodResolver(Resolver[T], Generic[T]):

    def __call__(self, *args, **kwargs) -> T:
        types = tuple([type(arg) for arg in args] + [type(kwarg) for kwarg in kwargs.values()])[1:]
        if types in self._dispatcher._functions:
            return self._dispatcher._functions[types](*args, **kwargs)
        else:
            raise TypeError(f"Not valid for type {str(types)} for function {self._dispatcher.name}")


def dispatcher(*types: type) -> Callable[[Callable[... , T]], Resolver[T]]:
    def wrapper(func: Callable[..., T]) -> Resolver[T]:
        name = ".".join([func.__module__, func.__qualname__])
        return Dispatcher.get(name)(func, types)
    return wrapper


def dispatch(func: Callable[..., T]) -> Resolver[T]:
    """
    Decorator for dispatcher.
    Args:
        func(``(...) -> T``): function to be dispatched.

    Returns:
        :class:`~tensorneko.util.dispatcher.Resolver[T]`: Resolver object.

    Example::

        @dispatch
        def add(x: int, y: int) -> int:
            return x + y

        @dispatch
        def add(x: List[int], y: List[int]) -> List[int]:
            assert len(x) == len(y)
            return [x[i] + y[i] for i in range(len(x))]

        add(1, 2)  # get 3
        add([1, 2], [3, 4])  # get [4, 6]

    """
    name = ".".join([func.__module__, func.__qualname__])
    return Dispatcher.get(name)(func)
