class Registry:
    """
    A registry class for easily creating a decorator-based register system.

    Examples::

        from tensorneko.util.registry import Registry
        model_registry = Registry()

        @model_registry.register("test")
        class ModelA:
            pass

        @model_registry.register("test2")
        class ModelB:
            pass

        @model_registry.register("test3")
        def test3():
            return "Test3"

        print(model_registry["test"]())
    """

    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        def wrapper(cls):
            self._registry[name] = cls
            return cls

        return wrapper

    def __getitem__(self, name: str):
        return self._registry[name]
