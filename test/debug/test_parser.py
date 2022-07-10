import sys
import unittest
from argparse import ArgumentParser

from tensorneko_util.debug import get_parser_default_args
import tensorneko_util


class ArgumentParserTest(unittest.TestCase):

    def test_parser_with_simple_default_value(self):
        parser = ArgumentParser()
        parser.add_argument("integers", type=int, nargs="+", default=[1, 2, 3])
        parser.add_argument("--sum", dest="accumulate", action="store_const", const=sum, default=max)
        args = get_parser_default_args(parser)
        self.assertEqual(args.integers, [1, 2, 3])
        self.assertEqual(args.accumulate, max)

    def test_parser_without_default_value(self):
        parser = ArgumentParser()
        parser.add_argument("integers", type=int, nargs="+")
        parser.add_argument("--sum", dest="accumulate", action="store_const", const=sum)
        args = get_parser_default_args(parser)
        self.assertEqual(args.integers, None)
        self.assertEqual(args.accumulate, None)

    def test_parser_with_bool_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--flag1", action="store_true")
        parser.add_argument("--flag2", action="store_false")
        args = get_parser_default_args(parser)
        self.assertEqual(args.flag1, False)
        self.assertEqual(args.flag2, True)

    def test_parser_with_const_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--const", action="store_const", const=1)
        args = get_parser_default_args(parser)
        self.assertEqual(args.const, None)

    def test_parser_with_append_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--append", action="append")
        args = get_parser_default_args(parser)
        self.assertEqual(args.append, None)

    def test_parser_with_append_const_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--append", action="append_const", const=1)
        args = get_parser_default_args(parser)
        self.assertEqual(args.append, None)

    def test_parser_with_count_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--count", action="count")
        args = get_parser_default_args(parser)
        self.assertEqual(args.count, None)

    def test_parser_with_default_count_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--count", action="count", default=0)
        args = get_parser_default_args(parser)
        self.assertEqual(args.count, 0)

    def test_parser_with_help_arg(self):
        parser = ArgumentParser()
        args = get_parser_default_args(parser)
        self.assertTrue("help" not in args.__dict__)

    def test_parser_with_version_arg(self):
        parser = ArgumentParser()
        parser.add_argument("--version", action="version", version=tensorneko_util.__version__)
        args = get_parser_default_args(parser)
        self.assertTrue("version" not in args.__dict__)

    def test_parser_with_extend_arg(self):
        if sys.version_info[1] < 8:
            return
        parser = ArgumentParser()
        parser.add_argument("--extend", action="extend", nargs="+", type=str)
        args = get_parser_default_args(parser)
        self.assertEqual(args.extend, None)
