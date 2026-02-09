import unittest
from unittest.mock import MagicMock, patch

import tensorneko_util.backend.parallel as parallel_backend
from tensorneko_util.backend.parallel import ExecutorPool, ParallelType, set_num_workers


class TestParallelBackend(unittest.TestCase):
    def setUp(self):
        self._original_num_workers = parallel_backend._num_workers
        ExecutorPool.shutdown_all()

    def tearDown(self):
        ExecutorPool.shutdown_all()
        parallel_backend._num_workers = self._original_num_workers

    def test_parallel_type_enum_values(self):
        self.assertEqual(ParallelType.THREAD.value, "thread")
        self.assertEqual(ParallelType.PROCESS.value, "process")

    def test_set_num_workers_updates_global(self):
        set_num_workers(3)
        self.assertEqual(parallel_backend._num_workers, 3)

    def test_get_thread_executor_reuses_named_pool(self):
        first = ExecutorPool.get_thread_executor("pool-a", num_workers=1)
        second = ExecutorPool.get_thread_executor("pool-a", num_workers=8)
        self.assertIs(first, second)

    def test_get_returns_thread_executor_for_thread_type(self):
        from_get = ExecutorPool.get(
            "pool-b", num_workers=1, parallel_type=ParallelType.THREAD
        )
        direct = ExecutorPool.get_thread_executor("pool-b", num_workers=1)
        self.assertIs(from_get, direct)

    def test_get_raises_for_invalid_parallel_type(self):
        with self.assertRaises(ValueError) as ctx:
            ExecutorPool.get("pool-c", parallel_type="invalid")
        self.assertIn("Invalid parallel type", str(ctx.exception))

    def test_submit_thread_executes_callable(self):
        future = ExecutorPool.submit(
            lambda x, y: x + y,
            2,
            3,
            name="submit-thread",
            parallel_type=ParallelType.THREAD,
        )
        self.assertEqual(future.result(timeout=1), 5)

    def test_class_getitem_string_returns_thread_pool(self):
        pool = ExecutorPool.get_thread_executor("indexed", num_workers=1)
        self.assertIs(ExecutorPool["indexed"], pool)

    def test_class_getitem_string_returns_process_pool(self):
        """Cover parallel.py line 56: process_pools[item] lookup by string."""
        fake_pool = MagicMock()
        ExecutorPool.process_pools["proc-only"] = fake_pool
        self.assertIs(ExecutorPool["proc-only"], fake_pool)

    def test_class_getitem_string_raises_when_ambiguous(self):
        name = "same-name"
        ExecutorPool.thread_pools[name] = MagicMock()
        ExecutorPool.process_pools[name] = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            _ = ExecutorPool[name]
        self.assertIn("Ambiguous indexing", str(ctx.exception))

    def test_class_getitem_string_raises_when_not_found(self):
        with self.assertRaises(ValueError) as ctx:
            _ = ExecutorPool["missing-pool"]
        self.assertIn("not found", str(ctx.exception))

    def test_class_getitem_tuple_thread_returns_pool(self):
        pool = ExecutorPool[("tuple-thread", ParallelType.THREAD)]
        self.assertIs(pool, ExecutorPool.thread_pools["tuple-thread"])

    def test_class_getitem_tuple_process_uses_getter_without_spawning(self):
        fake_executor = MagicMock()
        with patch.object(
            ExecutorPool, "get_process_executor", return_value=fake_executor
        ) as mock_get:
            pool = ExecutorPool[("tuple-process", ParallelType.PROCESS)]
        self.assertIs(pool, fake_executor)
        mock_get.assert_called_once_with("tuple-process")

    def test_class_getitem_tuple_raises_for_invalid_length(self):
        with self.assertRaises(ValueError) as ctx:
            _ = ExecutorPool[("name-only",)]
        self.assertIn("Invalid indexing", str(ctx.exception))

    def test_class_getitem_tuple_raises_for_invalid_parallel_type(self):
        with self.assertRaises(ValueError) as ctx:
            _ = ExecutorPool[("name", "invalid")]
        self.assertIn("Invalid parallel type", str(ctx.exception))

    def test_class_getitem_raises_for_invalid_item_type(self):
        with self.assertRaises(ValueError) as ctx:
            _ = ExecutorPool[123]
        self.assertIn("Invalid indexing", str(ctx.exception))

    def test_shutdown_thread_success_and_missing_error(self):
        ExecutorPool.get_thread_executor("shutdown-thread", num_workers=1)
        ExecutorPool.shutdown("shutdown-thread", parallel_type=ParallelType.THREAD)
        self.assertNotIn("shutdown-thread", ExecutorPool.thread_pools)

        with self.assertRaises(ValueError) as ctx:
            ExecutorPool.shutdown("shutdown-thread", parallel_type=ParallelType.THREAD)
        self.assertIn("Thread pool", str(ctx.exception))

    def test_shutdown_process_success_missing_and_invalid_type(self):
        process_pool = MagicMock()
        ExecutorPool.process_pools["shutdown-process"] = process_pool

        ExecutorPool.shutdown("shutdown-process", parallel_type=ParallelType.PROCESS)
        process_pool.shutdown.assert_called_once_with()
        self.assertNotIn("shutdown-process", ExecutorPool.process_pools)

        with self.assertRaises(ValueError) as ctx_missing:
            ExecutorPool.shutdown(
                "shutdown-process", parallel_type=ParallelType.PROCESS
            )
        self.assertIn("Process pool", str(ctx_missing.exception))

        with self.assertRaises(ValueError) as ctx_invalid:
            ExecutorPool.shutdown("anything", parallel_type="invalid")
        self.assertIn("Invalid parallel type", str(ctx_invalid.exception))

    def test_shutdown_all_clears_all_pools(self):
        ExecutorPool.get_thread_executor("all-thread", num_workers=1)
        process_pool = MagicMock()
        ExecutorPool.process_pools["all-process"] = process_pool

        ExecutorPool.shutdown_all()

        process_pool.shutdown.assert_called_once_with()
        self.assertEqual(ExecutorPool.thread_pools, {})
        self.assertEqual(ExecutorPool.process_pools, {})


if __name__ == "__main__":
    unittest.main()
