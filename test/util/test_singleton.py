import unittest

from tensorneko_util.util import Singleton


@Singleton
class MyObject:
    class_value = 1

    def __init__(self):
        self.value = 0

    def add(self, value):
        self.value += value

    @classmethod
    def add_class_value(cls, value):
        cls.class_value += value

    @staticmethod
    def static_method():
        return "static_method"


class SingletonTest(unittest.TestCase):

    def test_singleton_access_object(self):
        self.assertTrue(type(MyObject) is not type)

    def test_singleton_access_instance_attributes(self):
        self.assertEqual(MyObject.value, 0)

    def test_singleton_access_instance_methods(self):
        MyObject.add(1)
        self.assertEqual(MyObject.value, 1)

        MyObject.add(2)
        self.assertEqual(MyObject.value, 3)

    def test_singleton_access_class_attributes(self):
        self.assertEqual(MyObject.class_value, 1)

    def test_singleton_access_class_method(self):
        MyObject.add_class_value(1)
        self.assertEqual(MyObject.class_value, 2)
        MyObject.add_class_value(1)
        self.assertEqual(MyObject.class_value, 3)

    def test_singleton_access_static_method(self):
        self.assertEqual(MyObject.static_method(), "static_method")
