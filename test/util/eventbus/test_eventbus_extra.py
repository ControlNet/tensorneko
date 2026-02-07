"""Extra EventBus tests to cover remaining miss lines in bus.py."""

import asyncio
import unittest
import warnings

from tensorneko_util.util.eventbus.bus import (
    EventBus,
    EventHandler,
    EmitType,
    Listener,
)
from tensorneko_util.util.eventbus.event import Event


# ── Unique event types for this test module ──────────────────────────


class ThreadTestEvent(Event):
    def __init__(self, value: int):
        self.value = value


class AsyncTestEvent(Event):
    def __init__(self, value: int):
        self.value = value


class SubscribeProcessEvent(Event):
    def __init__(self, value: int):
        self.value = value


class NoAnnotationEvent(Event):
    def __init__(self):
        pass


class NotHandlerClass:
    """Not an EventHandler subclass."""

    def __call__(self, event: NoAnnotationEvent):
        pass


# ── Tests ────────────────────────────────────────────────────────────


class TestEventBusSubscribeErrors(unittest.TestCase):
    """Cover error paths in EventBus.subscribe."""

    def test_subscribe_non_event_handler_class_raises(self):
        """Line 240: subscribe a class that is not EventHandler subclass."""
        bus = EventBus()
        with self.assertRaises(ValueError) as ctx:
            bus.subscribe(NotHandlerClass, emit_type=EmitType.NORMAL)
        self.assertIn("not inherited from EventHandler", str(ctx.exception))

    def test_subscribe_no_annotation_raises(self):
        """Line 251: subscribe a function without 'event' annotation."""
        bus = EventBus()

        def handler_no_annotation(e):
            pass

        with self.assertRaises(TypeError) as ctx:
            bus.subscribe(handler_no_annotation)
        self.assertIn("annotated", str(ctx.exception))


class TestEventBusSubscribeProcess(unittest.TestCase):
    """Cover subscribe_process (line 266)."""

    def test_subscribe_process_registers(self):
        """Line 266: subscribe_process calls subscribe with PROCESS emit type."""
        bus = EventBus()
        results = []

        def handler(event: SubscribeProcessEvent):
            results.append(event.value)

        bus.subscribe_process(handler)
        # Verify registration
        self.assertIn(SubscribeProcessEvent, bus.listeners)
        listener = list(bus.listeners[SubscribeProcessEvent])[0]
        self.assertEqual(listener.emit_type, EmitType.PROCESS)


class TestEventBusRegisterFuncName(unittest.TestCase):
    """Cover register with func_name=None (line 179)."""

    def test_register_default_func_name(self):
        """Line 179: func_name defaults to func.__name__."""
        bus = EventBus()

        def my_handler(event: ThreadTestEvent):
            pass

        bus.register(
            ThreadTestEvent, my_handler, func_name=None, emit_type=EmitType.NORMAL
        )
        listener = list(bus.listeners[ThreadTestEvent])[0]
        self.assertEqual(listener.func_name, "my_handler")


class TestEventBusThreadEmit(unittest.TestCase):
    """Cover thread emit path (line 196-197, 209-210)."""

    def test_thread_emit(self):
        """Emit with THREAD listener."""
        bus = EventBus()
        results = []

        def handler(event: ThreadTestEvent):
            results.append(event.value)

        bus.subscribe_thread(handler)
        # Manually emit (not via Event metaclass to use our custom bus)
        event = object.__new__(ThreadTestEvent)
        event.value = 42
        bus.emit(event)
        bus.wait()
        self.assertEqual(results, [42])


class TestEventBusAsyncEmit(unittest.TestCase):
    """Cover async emit path (lines 200-201, 211-212)."""

    def test_async_emit(self):
        """Emit with ASYNC listener."""
        bus = EventBus()
        results = []

        async def handler(event: AsyncTestEvent):
            results.append(event.value)

        bus.subscribe_async(handler)
        event = object.__new__(AsyncTestEvent)
        event.value = 99
        bus.emit(event)
        self.assertEqual(results, [99])


class TestListenerDuplicateProcessWarning(unittest.TestCase):
    """Cover Listener.__post_init__ duplicate PROCESS warning (line 35)."""

    def setUp(self):
        self._orig_names = Listener.func_names.copy()

    def tearDown(self):
        Listener.func_names = self._orig_names

    def test_duplicate_process_listener_warns(self):
        """Line 35: registering same PROCESS func.__name__ warns."""

        def dup_proc_handler(event):
            pass

        # Add func.__name__ to func_names first
        Listener.func_names.add("dup_proc_handler")
        # Second registration with same func.__name__ should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Listener(dup_proc_handler, "dup_proc_handler", EmitType.PROCESS)
            process_warnings = [x for x in w if "already registered" in str(x.message)]
            self.assertTrue(len(process_warnings) > 0)


class TestEventSubclassBusAnnotation(unittest.TestCase):
    """Cover Event.__init_subclass__ bus annotation check (line 20)."""

    def test_bus_annotation_raises(self):
        """Line 20: subclass with 'bus' annotation in __init__ raises TypeError."""
        with self.assertRaises(TypeError) as ctx:

            class BadEvent(Event):
                def __init__(self, bus: EventBus):
                    pass

        self.assertIn("bus", str(ctx.exception))


class TestSubscribeDecoratorProcess(unittest.TestCase):
    """Cover subscribe.process decorator (decorator.py line 19)."""

    def test_subscribe_process_decorator(self):
        """decorator.py line 19: subscribe.process registers with PROCESS emit type."""
        from tensorneko_util.util.eventbus.decorator import subscribe as sub_dec

        class ProcDecEvent(Event):
            def __init__(self, val: int):
                self.val = val

        # Use a fresh bus to avoid polluting default
        original_default = EventBus.default
        test_bus = EventBus()
        EventBus.default = test_bus
        try:

            @sub_dec.process
            def proc_handler(event: ProcDecEvent):
                pass

            self.assertIn(ProcDecEvent, test_bus.listeners)
            listener = list(test_bus.listeners[ProcDecEvent])[0]
            self.assertEqual(listener.emit_type, EmitType.PROCESS)
        finally:
            EventBus.default = original_default


if __name__ == "__main__":
    unittest.main()
