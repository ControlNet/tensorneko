import unittest

import tensorneko


class TestLibraryInfo(unittest.TestCase):

    def test_version(self):
        version = tensorneko.io.read.text.of("version.txt")
        self.assertEqual(tensorneko.__version__, version)


if __name__ == '__main__':
    unittest.main()
