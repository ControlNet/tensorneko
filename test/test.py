import unittest

from tensorneko.hello import return_hello


class MyTestCase(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(return_hello(), "Hello TensorNeko")


if __name__ == '__main__':
    unittest.main()
