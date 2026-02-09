import subprocess
import unittest
from unittest.mock import patch, MagicMock

from tensorneko_util.visualization.tensorboard import Server


class TestTensorboardServer(unittest.TestCase):
    @patch("tensorneko_util.visualization.tensorboard.subprocess.Popen")
    def test_start_stop(self, mock_popen):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        server = Server(logdir="test_logs", port=6007)
        server.start()

        mock_popen.assert_called_once_with(
            ["tensorboard", "--logdir", "test_logs", "--port", "6007"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        self.assertIs(server.process, mock_proc)

        server.stop()
        mock_proc.terminate.assert_called_once()
        self.assertIsNone(server.process)

    @patch("tensorneko_util.visualization.tensorboard.subprocess.Popen")
    def test_context_manager(self, mock_popen):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        with Server(logdir="logs", port=6008) as srv:
            self.assertIs(srv.process, mock_proc)

        mock_proc.terminate.assert_called_once()

    @patch("tensorneko_util.visualization.tensorboard.subprocess.Popen")
    def test_custom_logdir_and_port(self, mock_popen):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        server = Server(logdir="my/custom/logs", port=7777)
        self.assertEqual(server.logdir, "my/custom/logs")
        self.assertEqual(server.port, 7777)

        server.start()
        args = mock_popen.call_args[0][0]
        self.assertIn("my/custom/logs", args)
        self.assertIn("7777", args)
        server.stop()

    @patch("tensorneko_util.visualization.tensorboard.subprocess.Popen")
    def test_default_values(self, mock_popen):
        server = Server()
        self.assertEqual(server.logdir, "logs")
        self.assertEqual(server.port, 6006)
        self.assertIsNone(server.process)

    @patch("tensorneko_util.visualization.tensorboard.subprocess.Popen")
    def test_stop_when_not_started(self, mock_popen):
        """Stopping a non-started server should not raise."""
        server = Server()
        server.stop()  # should just print "Server is stopped."

    @patch("tensorneko_util.visualization.tensorboard.subprocess.Popen")
    def test_restart(self, mock_popen):
        """Starting again should stop the old process first."""
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()
        mock_popen.side_effect = [mock_proc1, mock_proc2]

        server = Server(port=6009)
        server.start()
        self.assertIs(server.process, mock_proc1)

        server.start()  # restart
        mock_proc1.terminate.assert_called_once()
        self.assertIs(server.process, mock_proc2)
        server.stop()

    @patch("tensorneko_util.visualization.tensorboard.subprocess.run")
    def test_start_blocking(self, mock_run):
        """Lines 26-31: start_blocking uses subprocess.run."""
        mock_run.return_value = MagicMock()
        server = Server(logdir="blocking_logs", port=6010)
        server.start_blocking()
        mock_run.assert_called_once_with(
            ["tensorboard", "--logdir", "blocking_logs", "--port", "6010"]
        )

    @patch("tensorneko_util.visualization.tensorboard.subprocess.run")
    def test_start_blocking_restart(self, mock_run):
        """Lines 26-27: start_blocking stops old process first."""
        mock_run.return_value = MagicMock()
        server = Server(logdir="logs", port=6011)
        # Simulate an existing process
        mock_old_proc = MagicMock()
        server.process = mock_old_proc
        server.start_blocking()
        mock_old_proc.terminate.assert_called_once()
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
