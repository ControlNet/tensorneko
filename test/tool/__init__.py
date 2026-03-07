import unittest


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    _ = tests
    return loader.discover(start_dir=__path__[0], pattern=pattern or "test*.py")
