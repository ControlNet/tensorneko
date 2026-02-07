import subprocess
import sys
import unittest
from concurrent.futures import Future

from tensorneko_util.backend.blocking import run_blocking


class TestRunBlocking(unittest.TestCase):
    def test_run_blocking_with_future_returns_result(self):
        future = Future()
        future.set_result("done")

        @run_blocking
        def task():
            return future

        self.assertEqual(task(), "done")

    def test_run_blocking_with_popen_returns_returncode(self):
        @run_blocking
        def task():
            return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])

        self.assertEqual(task(), 7)

    def test_run_blocking_raises_type_error_for_unsupported_task_type(self):
        @run_blocking
        def task():
            return object()

        with self.assertRaises(TypeError) as ctx:
            task()
        self.assertIn("Future or Popen", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
