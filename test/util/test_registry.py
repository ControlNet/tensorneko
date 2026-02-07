import unittest

from tensorneko_util.util.registry import Registry


class UtilRegistryTest(unittest.TestCase):
    def test_registry_creation_is_empty(self):
        registry = Registry()
        self.assertEqual(list(registry.items()), [])
        self.assertEqual(registry._registry, {})

    def test_register_decorator_registers_class(self):
        registry = Registry()

        @registry.register("model")
        class Model:
            pass

        self.assertIs(registry["model"], Model)

    def test_register_decorator_registers_function(self):
        registry = Registry()

        @registry.register("builder")
        def build_value():
            return 42

        self.assertIs(registry["builder"], build_value)
        self.assertEqual(registry["builder"](), 42)

    def test_getitem_missing_name_raises_key_error(self):
        registry = Registry()
        with self.assertRaises(KeyError):
            _ = registry["missing"]

    def test_contains_and_items_for_registered_name(self):
        registry = Registry()

        @registry.register("x")
        class X:
            pass

        self.assertIn("x", registry._registry)
        self.assertEqual(dict(registry.items())["x"], X)

    def test_register_same_name_overwrites_previous_item(self):
        registry = Registry()

        @registry.register("key")
        class A:
            pass

        @registry.register("key")
        class B:
            pass

        self.assertIs(registry["key"], B)
