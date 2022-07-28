from typing import Type, Dict

from .type import T


class Singleton:
    """
    Singleton decorator. It is used to define the class as a singleton.

    Note: The constructor should not include any arguments.

    Example::

        from tensorneko.util import Singleton

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

    """

    all_instances: Dict[Type, object] = {}

    def __new__(cls, clazz: Type[T]) -> T:
        cls.all_instances[clazz] = clazz()
        return cls.all_instances[clazz]
