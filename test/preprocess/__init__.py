def load_tests(loader, tests, pattern):
    return loader.discover(start_dir=__path__[0], pattern="test*.py")
