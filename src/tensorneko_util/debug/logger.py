from __future__ import annotations

from logging import Logger


class DummyLogger(Logger):
    """
    A dummy logger that prints all messages to stdout. Or choose to not print anything.
    It is useful when you want to disable logging in a library.

    Args:
        verbose (``bool``): Whether to print the message to stdout.

    Examples::

        >>> import logging
        >>> from tensorneko_util.debug import DummyLogger
        >>> logger = logging.getLogger("annoy_logger", verbose=False)
        >>> DummyLogger.convert(logger)
        >>> logger.info("hello info")
        >>> logger.debug("hello debug")
        >>> logger.warning("hello warning")
        >>> # no output

    """

    def __init__(self, verbose: bool):
        self.verbose = verbose

    @classmethod
    def convert(cls, logger: Logger, verbose: bool = False) -> None:
        """Build a dummy logger from a real logger. It's an inplace operation."""
        cls.__init__(logger, verbose)
        logger.__class__ = cls

    def debug(
        self,
        msg,
        *args,
        exc_info=...,
        stack_info=...,
        stacklevel=...,
        extra=...,
    ):
        if self.verbose:
            print(msg)

    def info(
        self,
        msg,
        *args,
        exc_info=...,
        stack_info=...,
        stacklevel=...,
        extra=...,
    ):
        if self.verbose:
            print(msg)

    def warning(
        self,
        msg,
        *args,
        exc_info=...,
        stack_info=...,
        stacklevel=...,
        extra=...,
    ):
        if self.verbose:
            print(msg)

    def error(
        self,
        msg,
        *args,
        exc_info=...,
        stack_info=...,
        stacklevel=...,
        extra=...,
    ):
        if self.verbose:
            print(msg)

    def critical(
        self,
        msg,
        *args,
        exc_info=...,
        stack_info=...,
        stacklevel=...,
        extra=...,
    ):
        if self.verbose:
            print(msg)
