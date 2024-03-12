from typing import Type, Dict, Callable

from .type import T


class Singleton:
    """
    Singleton decorator. It is used to define the class as a singleton.

    Example::

        from tensorneko.util import Singleton

        # the singleton without args
        @Singleton
        class MyObject:
            def __init__(self):
                self.value = 0

            def add(self, value):
                self.value += value
                return self.value


        print(MyObject.value)  # 0
        MyObject.add(1)
        print(MyObject.value)  # 1

        # the singleton with args
        @Singleton.args(1, 2)
        class MyObject:
            def __init__(self, a, b):
                self.value = a + b

            def add(self, value):
                self.value += value
                return self.value

        print(MyObject.value)  # 3
        MyObject.add(1)
        print(MyObject.value) # 4

    """

    all_instances: Dict[Type, object] = {}

    def __new__(cls, clazz: Type[T]) -> T:
        cls.all_instances[clazz] = clazz()
        return cls.all_instances[clazz]

    @classmethod
    def args(cls, *args, **kwargs) -> Callable[[Type[T]], T]:
        def wrapper(clazz: Type[T]) -> T:
            cls.all_instances[clazz] = clazz(*args, **kwargs)
            return cls.all_instances[clazz]
        return wrapper
