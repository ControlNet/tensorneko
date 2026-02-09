import unittest


def load_tests(loader, _tests, _pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromName("test.msg.test_gotify"))
    suite.addTests(loader.loadTestsFromName("test.msg.test_postgres"))
    return suite
