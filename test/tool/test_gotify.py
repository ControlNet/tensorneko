import argparse
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

import tensorneko_tool.gotify as gotify_module


class TestRegisterSubparser(unittest.TestCase):
    def test_register_subparser_adds_expected_arguments(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        gotify_module.register_subparser(subparsers)
        gotify_parser = subparsers.choices["gotify"]

        action_dests = {action.dest for action in gotify_parser._actions}
        self.assertTrue(
            {
                "message",
                "title",
                "priority",
                "url",
                "token",
                "watch",
                "interval",
            }.issubset(action_dests)
        )

    def test_register_subparser_parses_defaults_and_custom_values(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        gotify_module.register_subparser(subparsers)

        args_default = parser.parse_args(["gotify", "hello"])
        self.assertEqual(args_default.message, "hello")
        self.assertEqual(args_default.priority, 0)
        self.assertEqual(args_default.interval, 5.0)
        self.assertIs(args_default.func, gotify_module.run)

        args_custom = parser.parse_args(
            [
                "gotify",
                "done",
                "--title",
                "my-title",
                "--priority",
                "3",
                "--url",
                "https://gotify.local",
                "--token",
                "abc",
                "--watch",
                "worker",
                "--interval",
                "1.5",
            ]
        )
        self.assertEqual(args_custom.title, "my-title")
        self.assertEqual(args_custom.priority, 3)
        self.assertEqual(args_custom.url, "https://gotify.local")
        self.assertEqual(args_custom.token, "abc")
        self.assertEqual(args_custom.watch, "worker")
        self.assertEqual(args_custom.interval, 1.5)


class TestRun(unittest.TestCase):
    @patch("tensorneko_tool.gotify._run_watch", return_value=17)
    def test_run_dispatches_to_watch_with_pid(self, mock_run_watch):
        args = Namespace(message="m", watch="123", interval=0.2)

        result = gotify_module.run(args)

        self.assertEqual(result, 17)
        mock_run_watch.assert_called_once_with(args, None, 123)

    @patch("tensorneko_tool.gotify._run_watch", return_value=19)
    def test_run_dispatches_to_watch_with_name(self, mock_run_watch):
        args = Namespace(message="m", watch="worker", interval=0.2)

        result = gotify_module.run(args)

        self.assertEqual(result, 19)
        mock_run_watch.assert_called_once_with(args, "worker", None)

    @patch("tensorneko_tool.gotify.utils.print_error")
    def test_run_returns_two_when_message_missing_and_not_watching(
        self, mock_print_error
    ):
        args = Namespace(message=None, watch=None)

        result = gotify_module.run(args)

        self.assertEqual(result, 2)
        mock_print_error.assert_called_once_with(
            "Message is required when not using --watch"
        )

    @patch("tensorneko_tool.gotify._run_send", return_value=0)
    def test_run_dispatches_to_send_when_message_present(self, mock_run_send):
        args = Namespace(message="hello", watch=None)

        result = gotify_module.run(args)

        self.assertEqual(result, 0)
        mock_run_send.assert_called_once_with(args)


class TestRunSend(unittest.TestCase):
    @patch("tensorneko_tool.gotify.gotify_client.push")
    @patch("tensorneko_tool.gotify.utils.make_panel", return_value="panel")
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_send_success_with_explicit_title_and_url(
        self, mock_console, mock_make_panel, mock_push
    ):
        mock_console.status.return_value.__enter__.return_value = None
        mock_console.status.return_value.__exit__.return_value = False
        args = Namespace(
            message="finished",
            url="https://gotify.local",
            token="token",
            title="build",
            priority=2,
        )

        result = gotify_module._run_send(args)

        self.assertEqual(result, 0)
        mock_push.assert_called_once_with(
            "finished", "https://gotify.local", "token", "build", 2
        )
        body = mock_make_panel.call_args[0][0]
        self.assertIn("Title: build", body)
        self.assertIn("Server: https://gotify.local", body)
        self.assertIn("Priority: 2", body)
        self.assertIn("Message: finished", body)
        mock_console.print.assert_called_once_with("panel")

    @patch("tensorneko_tool.gotify.gotify_client.push")
    @patch("tensorneko_tool.gotify.utils.make_panel", return_value="panel")
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_send_success_uses_environment_title_and_url(
        self, mock_console, mock_make_panel, mock_push
    ):
        mock_console.status.return_value.__enter__.return_value = None
        mock_console.status.return_value.__exit__.return_value = False
        args = Namespace(
            message="from-env", url=None, token="token", title=None, priority=1
        )

        with patch.dict(
            "tensorneko_tool.gotify.os.environ",
            {"GOTIFY_TITLE": "env-title", "GOTIFY_URL": "https://env-gotify"},
            clear=True,
        ):
            result = gotify_module._run_send(args)

        self.assertEqual(result, 0)
        mock_push.assert_called_once_with("from-env", None, "token", None, 1)
        body = mock_make_panel.call_args[0][0]
        self.assertIn("Title: env-title", body)
        self.assertIn("Server: https://env-gotify", body)

    @patch("tensorneko_tool.gotify.gotify_client.push")
    @patch("tensorneko_tool.gotify.utils.make_panel", return_value="panel")
    @patch("tensorneko_tool.gotify.socket.gethostname", return_value="host-1")
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_send_success_uses_hostname_and_unknown_server(
        self, mock_console, _mock_hostname, mock_make_panel, mock_push
    ):
        mock_console.status.return_value.__enter__.return_value = None
        mock_console.status.return_value.__exit__.return_value = False
        args = Namespace(
            message="fallback", url=None, token="token", title=None, priority=0
        )

        with patch.dict("tensorneko_tool.gotify.os.environ", {}, clear=True):
            result = gotify_module._run_send(args)

        self.assertEqual(result, 0)
        mock_push.assert_called_once_with("fallback", None, "token", None, 0)
        body = mock_make_panel.call_args[0][0]
        self.assertIn("Title: host-1", body)
        self.assertIn("Server: Unknown", body)

    @patch("tensorneko_tool.gotify.utils.print_error")
    @patch(
        "tensorneko_tool.gotify.gotify_client.push", side_effect=RuntimeError("boom")
    )
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_send_failure_returns_one_and_prints_error(
        self, mock_console, _mock_push, mock_print_error
    ):
        mock_console.status.return_value.__enter__.return_value = None
        mock_console.status.return_value.__exit__.return_value = False
        args = Namespace(message="failed", url="u", token="t", title="x", priority=9)

        result = gotify_module._run_send(args)

        self.assertEqual(result, 1)
        mock_print_error.assert_called_once_with("Failed to send Gotify message: boom")


class TestRunWatch(unittest.TestCase):
    @patch("tensorneko_tool.gotify._run_send", return_value=0)
    @patch("tensorneko_tool.gotify.time.sleep")
    @patch("tensorneko_tool.gotify._is_process_running", side_effect=[True, False])
    @patch("tensorneko_tool.gotify.utils.make_panel", return_value="panel")
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_watch_triggers_send_when_seen_then_stopped(
        self,
        mock_console,
        _mock_make_panel,
        mock_is_running,
        mock_sleep,
        mock_run_send,
    ):
        args = Namespace(
            message=None, interval=0.01, title="t", url="u", token="k", priority=2
        )

        result = gotify_module._run_watch(args, target_name="worker", target_pid=None)

        self.assertEqual(result, 0)
        self.assertEqual(mock_is_running.call_count, 2)
        mock_is_running.assert_any_call("worker", None)
        mock_sleep.assert_called_once_with(0.01)
        mock_console.print.assert_called_once_with("panel")
        sent_args = mock_run_send.call_args[0][0]
        self.assertEqual(sent_args.message, 'Process "worker" is finished')
        self.assertEqual(sent_args.title, "t")

    @patch("tensorneko_tool.gotify._run_send", return_value=23)
    @patch("tensorneko_tool.gotify.time.sleep")
    @patch("tensorneko_tool.gotify._is_process_running", side_effect=[True, False])
    @patch("tensorneko_tool.gotify.utils.make_panel", return_value="panel")
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_watch_uses_existing_message_when_present(
        self,
        _mock_console,
        _mock_make_panel,
        _mock_is_running,
        _mock_sleep,
        mock_run_send,
    ):
        args = Namespace(
            message="custom",
            interval=0.01,
            title=None,
            url=None,
            token=None,
            priority=0,
        )

        result = gotify_module._run_watch(args, target_name=None, target_pid=42)

        self.assertEqual(result, 23)
        sent_args = mock_run_send.call_args[0][0]
        self.assertEqual(sent_args.message, "custom")

    @patch("tensorneko_tool.gotify._run_send")
    @patch("tensorneko_tool.gotify.time.sleep")
    @patch("tensorneko_tool.gotify._is_process_running", side_effect=KeyboardInterrupt)
    @patch("tensorneko_tool.gotify.utils.make_panel", return_value="panel")
    @patch("tensorneko_tool.gotify.utils.console")
    def test_run_watch_keyboard_interrupt_returns_130(
        self,
        _mock_console,
        _mock_make_panel,
        _mock_is_running,
        mock_sleep,
        mock_run_send,
    ):
        args = Namespace(
            message="m", interval=0.01, title=None, url=None, token=None, priority=0
        )

        result = gotify_module._run_watch(args, target_name="worker", target_pid=None)

        self.assertEqual(result, 130)
        mock_sleep.assert_not_called()
        mock_run_send.assert_not_called()


class TestIsProcessRunning(unittest.TestCase):
    @patch("tensorneko_tool.gotify.psutil.Process")
    @patch("tensorneko_tool.gotify.psutil.pid_exists", return_value=True)
    def test_is_process_running_pid_true_when_exists_and_running(
        self, mock_pid_exists, mock_process
    ):
        mock_process.return_value.is_running.return_value = True

        result = gotify_module._is_process_running(name=None, pid=99)

        self.assertTrue(result)
        mock_pid_exists.assert_called_once_with(99)
        mock_process.assert_called_once_with(99)

    @patch("tensorneko_tool.gotify.psutil.Process")
    @patch("tensorneko_tool.gotify.psutil.pid_exists", return_value=False)
    def test_is_process_running_pid_false_when_pid_missing(
        self, mock_pid_exists, mock_process
    ):
        result = gotify_module._is_process_running(name=None, pid=77)

        self.assertFalse(result)
        mock_pid_exists.assert_called_once_with(77)
        mock_process.assert_not_called()

    @patch("tensorneko_tool.gotify.psutil.process_iter")
    def test_is_process_running_name_true_when_name_matches(self, mock_process_iter):
        proc = MagicMock()
        proc.info = {"name": "Python", "cmdline": ["-m", "module"]}
        mock_process_iter.return_value = [proc]

        result = gotify_module._is_process_running(name="python", pid=None)

        self.assertTrue(result)
        mock_process_iter.assert_called_once_with(["name", "cmdline"])

    @patch("tensorneko_tool.gotify.psutil.process_iter")
    def test_is_process_running_name_true_when_cmdline_contains_name(
        self, mock_process_iter
    ):
        proc = MagicMock()
        proc.info = {"name": "bash", "cmdline": ["/usr/bin/python", "worker.py"]}
        mock_process_iter.return_value = [proc]

        result = gotify_module._is_process_running(name="python", pid=None)

        self.assertTrue(result)

    @patch("tensorneko_tool.gotify.psutil.process_iter")
    def test_is_process_running_name_false_when_no_match(self, mock_process_iter):
        proc1 = MagicMock()
        proc1.info = {"name": "bash", "cmdline": ["/usr/bin/bash", "-lc"]}
        proc2 = MagicMock()
        proc2.info = {"name": "node", "cmdline": ["node", "server.js"]}
        mock_process_iter.return_value = [proc1, proc2]

        result = gotify_module._is_process_running(name="python", pid=None)

        self.assertFalse(result)

    def test_is_process_running_returns_false_when_name_and_pid_none(self):
        result = gotify_module._is_process_running(name=None, pid=None)
        self.assertFalse(result)


class TestOverrideArgs(unittest.TestCase):
    def test_override_args_overrides_selected_attributes_and_keeps_others(self):
        original = Namespace(
            message="old",
            title="title",
            priority=1,
            url="https://url",
            token="tk",
            watch="w",
            interval=1.0,
        )

        overridden = gotify_module._OverrideArgs(original, message="new", priority=5)

        self.assertEqual(overridden.message, "new")
        self.assertEqual(overridden.priority, 5)
        self.assertEqual(overridden.title, "title")
        self.assertEqual(overridden.url, "https://url")
        self.assertEqual(overridden.token, "tk")
        self.assertEqual(overridden.watch, "w")
        self.assertEqual(overridden.interval, 1.0)
        self.assertEqual(original.message, "old")
        self.assertEqual(original.priority, 1)


if __name__ == "__main__":
    unittest.main()
