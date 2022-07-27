import asyncio
import random
import time
import unittest

from tensorneko_util.util import subscribe, subscribe_thread, subscribe_async, Event
from tensorneko_util.util.eventbus.bus import EventHandler


class AddValueEvent(Event):

    def __init__(self, value: int):
        self.value = value


@subscribe
class CustomHandler(EventHandler):

    def __init__(self):
        self.x = 0

    def __call__(self, event: AddValueEvent):
        self.x += event.value


class AddValueThreadEvent(Event):

    def __init__(self, value: int):
        self.value = value


@subscribe_thread
class CustomThreadHandler(EventHandler):

    def __init__(self):
        self.x = 0

    def __call__(self, event: AddValueThreadEvent):
        time.sleep(random.random())
        self.x += event.value


class AddValueAsyncEvent(Event):

    def __init__(self, value: int):
        self.value = value


@subscribe_async
class CustomAsyncHandler(EventHandler):

    def __init__(self):
        self.x = 0

    async def __call__(self, event: AddValueAsyncEvent):
        await asyncio.sleep(random.random())
        self.x += event.value


class HandlerClassTest(unittest.TestCase):

    def test_custom_handler_class(self):
        AddValueEvent(10)
        AddValueEvent(20)
        self.assertEqual(CustomHandler.instance().x, 30)

    def test_custom_thread_handler_class(self):
        AddValueThreadEvent(10)
        AddValueThreadEvent(20)
        self.assertEqual(CustomThreadHandler.instance().x, 30)

    def test_custom_async_handler_class(self):
        AddValueAsyncEvent(10)
        AddValueAsyncEvent(20)
        self.assertEqual(CustomAsyncHandler.instance().x, 30)
