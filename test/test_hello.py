import unittest

from tensorneko.hello import return_hello


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(return_hello(), "Hello TensorNeko")
