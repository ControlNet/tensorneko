import contextlib
import logging
import unittest
from io import StringIO

from tensorneko_util.debug.logger import DummyLogger


class DummyLoggerTest(unittest.TestCase):
    @staticmethod
    def _build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        return logger

    def test_convert_to_dummy_logger_preserves_level(self):
        logger = self._build_logger("dummy_logger_convert", logging.WARNING)
        DummyLogger.convert(logger, verbose=False)

        self.assertIsInstance(logger, DummyLogger)
        self.assertEqual(logger.level, logging.WARNING)
        self.assertFalse(logger.verbose)

    def test_debug_prints_message_when_verbose(self):
        logger = self._build_logger("dummy_logger_debug", logging.DEBUG)
        DummyLogger.convert(logger, verbose=True)

        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            logger.debug("debug message")

        self.assertEqual(stream.getvalue(), "debug message\n")

    def test_all_log_methods_print_when_verbose(self):
        logger = self._build_logger("dummy_logger_all_methods", logging.INFO)
        DummyLogger.convert(logger, verbose=True)

        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            logger.info("custom-format: value=1")
            logger.warning("warn message")
            logger.error("error message")
            logger.critical("critical message")

        self.assertEqual(
            stream.getvalue().splitlines(),
            [
                "custom-format: value=1",
                "warn message",
                "error message",
                "critical message",
            ],
        )

    def test_no_output_for_all_methods_when_not_verbose(self):
        logger = self._build_logger("dummy_logger_silent", logging.DEBUG)
        DummyLogger.convert(logger, verbose=False)

        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            logger.debug("d")
            logger.info("i")
            logger.warning("w")
            logger.error("e")
            logger.critical("c")

        self.assertEqual(stream.getvalue(), "")

    def test_printing_does_not_depend_on_logger_level(self):
        logger = self._build_logger("dummy_logger_level", logging.CRITICAL)
        DummyLogger.convert(logger, verbose=True)

        stream = StringIO()
        with contextlib.redirect_stdout(stream):
            logger.debug("still printed")

        self.assertEqual(stream.getvalue(), "still printed\n")
