import asyncio
import io
import time
import random
import unittest
from unittest.mock import patch
from itertools import permutations

from tensorneko_util.util import subscribe, Event, EventBus


@subscribe
def normal_handler(event: Event):
    print("normal_handler is called")


class CustomEvent(Event):
    def __init__(self, x):
        self.x = x


@subscribe
def custom_handler(event: CustomEvent):
    print("custom_handler is called, x =", event.x)


class ThreadEvent(Event):
    def __init__(self, x):
        self.x = x


@subscribe
def thread_handler_normal(event: ThreadEvent):
    time.sleep(random.random())
    print("thread_handler_normal is called, x =", event.x)


@subscribe.thread
def thread_handler_thread(event: ThreadEvent):
    time.sleep(random.random())
    print("thread_handler_thread is called, x =", event.x)


@subscribe.thread
def thread_handler_thread(event: ThreadEvent):
    time.sleep(random.random())
    print("thread_handler_thread2 is called, x =", event.x)


class AsyncEvent(Event):
    def __init__(self, x):
        self.x = x


@subscribe.coro
async def async_handler_async(event: AsyncEvent):
    await asyncio.sleep(random.random())
    print("async_handler_async is called, x =", event.x)


@subscribe.coro
async def async_handler_async(event: AsyncEvent):
    await asyncio.sleep(random.random())
    print("async_handler_async2 is called, x =", event.x)


class MixedEvent(Event):
    def __init__(self, x):
        self.x = x


@subscribe
def mixed_handler_normal(event: MixedEvent):
    time.sleep(random.random())
    print("mixed_handler_normal is called, x =", event.x)


@subscribe.thread
def mixed_handler_thread(event: MixedEvent):
    time.sleep(random.random())
    print("mixed_handler_thread is called, x =", event.x)


@subscribe.coro
async def mixed_handler_async(event: MixedEvent):
    await asyncio.sleep(random.random())
    print("mixed_handler_async is called, x =", event.x)


class EventBusTest(unittest.TestCase):
    # `io.StringIO` cannot catch multiprocess output, so skip.

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_normal_once(self, mock_stdout):
        Event()
        self.assertEqual(mock_stdout.getvalue(), "normal_handler is called\n")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_normal_twice(self, mock_stdout):
        Event()
        Event()
        self.assertEqual(mock_stdout.getvalue(), "normal_handler is called\nnormal_handler is called\n")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_normal_custom_once(self, mock_stdout):
        CustomEvent(100)
        EventBus.default.wait()
        self.assertEqual(mock_stdout.getvalue(), "custom_handler is called, x = 100\n")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_normal_custom_twice(self, mock_stdout):
        CustomEvent(100)
        EventBus.default.wait()
        CustomEvent(200)
        EventBus.default.wait()
        self.assertEqual(mock_stdout.getvalue(), "custom_handler is called, x = 100\n"
                                                 "custom_handler is called, x = 200\n")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_thread_once(self, mock_stdout):
        ThreadEvent(100)
        EventBus.default.wait()
        all_possibilities = permutations([
            "thread_handler_normal is called, x = 100\n",
            "thread_handler_thread is called, x = 100\n",
            "thread_handler_thread2 is called, x = 100\n"
        ])
        self.assertIn(mock_stdout.getvalue(), ["".join(p) for p in all_possibilities])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_thread_twice(self, mock_stdout):
        ThreadEvent(100)
        EventBus.default.wait()
        ThreadEvent(200)
        EventBus.default.wait()
        all_possibilities_100 = ["".join(each) for each in permutations([
            "thread_handler_normal is called, x = 100\n",
            "thread_handler_thread is called, x = 100\n",
            "thread_handler_thread2 is called, x = 100\n"
        ])]

        all_possibilities_200 = ["".join(each) for each in permutations([
            "thread_handler_normal is called, x = 200\n",
            "thread_handler_thread is called, x = 200\n",
            "thread_handler_thread2 is called, x = 200\n"
        ])]

        all_p = [p + q for p in all_possibilities_100 for q in all_possibilities_200]
        self.assertIn(mock_stdout.getvalue(), all_p)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_async_once(self, mock_stdout):
        AsyncEvent(100)
        EventBus.default.wait()
        all_possibilities = permutations([
            "async_handler_async is called, x = 100\n",
            "async_handler_async2 is called, x = 100\n"
        ])

        self.assertIn(mock_stdout.getvalue(), ["".join(p) for p in all_possibilities])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_handle_async_twice(self, mock_stdout):
        AsyncEvent(100)
        EventBus.default.wait()
        AsyncEvent(200)
        EventBus.default.wait()
        all_possibilities_100 = ["".join(each) for each in permutations([
            "async_handler_async is called, x = 100\n",
            "async_handler_async2 is called, x = 100\n"
        ])]

        all_possibilities_200 = ["".join(each) for each in permutations([
            "async_handler_async is called, x = 200\n",
            "async_handler_async2 is called, x = 200\n"
        ])]

        all_p = [p + q for p in all_possibilities_100 for q in all_possibilities_200]
        self.assertIn(mock_stdout.getvalue(), all_p)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_all_mixed(self, mock_stdout):
        MixedEvent(100)
        EventBus.default.wait()
        all_possibilities = permutations([
            "mixed_handler_normal is called, x = 100\n",
            "mixed_handler_thread is called, x = 100\n",
            "mixed_handler_async is called, x = 100\n"
        ])

        self.assertIn(mock_stdout.getvalue(), ["".join(p) for p in all_possibilities])

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_all_mixed_twice(self, mock_stdout):
        MixedEvent(100)
        EventBus.default.wait()
        MixedEvent(200)
        EventBus.default.wait()
        all_possibilities_100 = ["".join(each) for each in permutations([
            "mixed_handler_normal is called, x = 100\n",
            "mixed_handler_thread is called, x = 100\n",
            "mixed_handler_async is called, x = 100\n"
        ])]

        all_possibilities_200 = ["".join(each) for each in permutations([
            "mixed_handler_normal is called, x = 200\n",
            "mixed_handler_thread is called, x = 200\n",
            "mixed_handler_async is called, x = 200\n"
        ])]

        all_p = [p + q for p in all_possibilities_100 for q in all_possibilities_200]
        self.assertIn(mock_stdout.getvalue(), all_p)
