import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

from tensorneko_tool import utils
from tensorneko_tool.utils import (
    get_tensorneko_tool_path,
    version,
    get_banner,
    print_banner,
    print_info,
    print_success,
    print_error,
    make_panel,
    set_quiet,
    QuietConsole,
    _NullStatus,
)
from rich.panel import Panel
from rich.console import Console


class TestGetTensornekoToolPath(unittest.TestCase):
    def test_get_tensorneko_tool_path_returns_string(self):
        """Test that get_tensorneko_tool_path returns a string"""
        result = get_tensorneko_tool_path()
        self.assertIsInstance(result, str)

    def test_get_tensorneko_tool_path_ends_with_tensorneko_tool(self):
        """Test that path ends with 'tensorneko_tool'"""
        result = get_tensorneko_tool_path()
        self.assertTrue(result.endswith("tensorneko_tool"))

    def test_get_tensorneko_tool_path_is_absolute(self):
        """Test that path is absolute"""
        result = get_tensorneko_tool_path()
        self.assertTrue(result.startswith("/"))


class TestVersion(unittest.TestCase):
    def test_version_is_string(self):
        """Test that version returns a string"""
        result = version.value
        self.assertIsInstance(result, str)

    def test_version_is_not_empty(self):
        """Test that version string is not empty"""
        result = version.value
        self.assertTrue(len(result) > 0)

    def test_version_matches_version_txt(self):
        """Test that version matches content of version.txt"""
        result = version.value
        with open(f"{get_tensorneko_tool_path()}/version.txt", "r") as file:
            expected = file.read().strip()
        self.assertEqual(result, expected)


class TestGetBanner(unittest.TestCase):
    def test_get_banner_returns_string(self):
        """Test that get_banner returns a string"""
        result = get_banner()
        self.assertIsInstance(result, str)

    def test_get_banner_contains_tensorneko(self):
        """Test that banner contains 'TensorNeko'"""
        result = get_banner()
        self.assertIn("TensorNeko", result)

    def test_get_banner_is_not_empty(self):
        """Test that banner is not empty"""
        result = get_banner()
        self.assertTrue(len(result) > 0)

    def test_get_banner_contains_github_url(self):
        """Test that banner contains GitHub URL"""
        result = get_banner()
        self.assertIn("github.com", result)


class TestPrintBanner(unittest.TestCase):
    def setUp(self):
        """Save original console for restoration"""
        self.original_console = utils.console

    def tearDown(self):
        """Restore original console"""
        utils.console = self.original_console

    @patch("tensorneko_tool.utils.console")
    def test_print_banner_calls_console_print(self, mock_console):
        """Test that print_banner calls console.print"""
        utils.console = mock_console
        print_banner()
        mock_console.print.assert_called_once()

    @patch("tensorneko_tool.utils.console")
    def test_print_banner_prints_success_styled_banner(self, mock_console):
        """Test that print_banner prints with [success] style"""
        utils.console = mock_console
        print_banner()
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("[success]", call_args)
        self.assertIn("[/success]", call_args)


class TestPrintInfo(unittest.TestCase):
    def setUp(self):
        """Save original console for restoration"""
        self.original_console = utils.console

    def tearDown(self):
        """Restore original console"""
        utils.console = self.original_console

    @patch("tensorneko_tool.utils.console")
    def test_print_info_calls_console_print(self, mock_console):
        """Test that print_info calls console.print"""
        utils.console = mock_console
        print_info("test message")
        mock_console.print.assert_called_once()

    @patch("tensorneko_tool.utils.console")
    def test_print_info_includes_message(self, mock_console):
        """Test that print_info includes the message"""
        utils.console = mock_console
        print_info("test message")
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("test message", call_args)

    @patch("tensorneko_tool.utils.console")
    def test_print_info_uses_info_style(self, mock_console):
        """Test that print_info uses [info] style"""
        utils.console = mock_console
        print_info("test message")
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("[info]", call_args)
        self.assertIn("[/info]", call_args)


class TestPrintSuccess(unittest.TestCase):
    def setUp(self):
        """Save original console for restoration"""
        self.original_console = utils.console

    def tearDown(self):
        """Restore original console"""
        utils.console = self.original_console

    @patch("tensorneko_tool.utils.console")
    def test_print_success_calls_console_print(self, mock_console):
        """Test that print_success calls console.print"""
        utils.console = mock_console
        print_success("test message")
        mock_console.print.assert_called_once()

    @patch("tensorneko_tool.utils.console")
    def test_print_success_includes_message(self, mock_console):
        """Test that print_success includes the message"""
        utils.console = mock_console
        print_success("test message")
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("test message", call_args)

    @patch("tensorneko_tool.utils.console")
    def test_print_success_uses_success_style(self, mock_console):
        """Test that print_success uses [success] style"""
        utils.console = mock_console
        print_success("test message")
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("[success]", call_args)
        self.assertIn("[/success]", call_args)


class TestPrintError(unittest.TestCase):
    def setUp(self):
        """Save original console for restoration"""
        self.original_console = utils.console

    def tearDown(self):
        """Restore original console"""
        utils.console = self.original_console

    @patch("tensorneko_tool.utils.console")
    def test_print_error_calls_console_print(self, mock_console):
        """Test that print_error calls console.print"""
        utils.console = mock_console
        print_error("test message")
        mock_console.print.assert_called_once()

    @patch("tensorneko_tool.utils.console")
    def test_print_error_includes_message(self, mock_console):
        """Test that print_error includes the message"""
        utils.console = mock_console
        print_error("test message")
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("test message", call_args)

    @patch("tensorneko_tool.utils.console")
    def test_print_error_uses_error_style(self, mock_console):
        """Test that print_error uses [error] style"""
        utils.console = mock_console
        print_error("test message")
        call_args = mock_console.print.call_args[0][0]
        self.assertIn("[error]", call_args)
        self.assertIn("[/error]", call_args)


class TestMakePanel(unittest.TestCase):
    def test_make_panel_returns_panel(self):
        """Test that make_panel returns a Panel object"""
        result = make_panel("body", "title")
        self.assertIsInstance(result, Panel)

    def test_make_panel_with_default_border_style(self):
        """Test make_panel with default border_style"""
        result = make_panel("body", "title")
        self.assertIsInstance(result, Panel)
        self.assertEqual(result.expand, True)

    def test_make_panel_with_custom_border_style(self):
        """Test make_panel with custom border_style"""
        result = make_panel("body", "title", border_style="warning")
        self.assertIsInstance(result, Panel)

    def test_make_panel_has_title(self):
        """Test that panel has title set"""
        result = make_panel("body", "test_title")
        self.assertIsNotNone(result.title)

    def test_make_panel_has_border_style(self):
        """Test that panel has border_style set"""
        result = make_panel("body", "title", border_style="info")
        self.assertEqual(result.border_style, "info")


class TestSetQuiet(unittest.TestCase):
    def setUp(self):
        """Save original console for restoration"""
        self.original_console = utils.console

    def tearDown(self):
        """Restore original console"""
        utils.console = self.original_console

    def test_set_quiet_true_replaces_console_with_quiet_console(self):
        """Test that set_quiet(True) replaces console with QuietConsole"""
        set_quiet(True)
        self.assertIsInstance(utils.console, QuietConsole)

    def test_set_quiet_false_restores_console(self):
        """Test that set_quiet(False) restores Console"""
        set_quiet(True)
        set_quiet(False)
        self.assertIsInstance(utils.console, Console)
        self.assertNotIsInstance(utils.console, QuietConsole)

    def test_set_quiet_cycle(self):
        """Test cycling set_quiet on and off"""
        original = utils.console
        set_quiet(True)
        self.assertIsInstance(utils.console, QuietConsole)
        set_quiet(False)
        self.assertIsInstance(utils.console, Console)


class TestQuietConsole(unittest.TestCase):
    def setUp(self):
        """Create QuietConsole instance"""
        self.console = QuietConsole()

    def test_quiet_console_print_returns_none(self):
        """Test that QuietConsole.print returns None"""
        result = self.console.print("test")
        self.assertIsNone(result)

    def test_quiet_console_print_accepts_args(self):
        """Test that QuietConsole.print accepts multiple args"""
        result = self.console.print("arg1", "arg2", "arg3")
        self.assertIsNone(result)

    def test_quiet_console_print_accepts_kwargs(self):
        """Test that QuietConsole.print accepts kwargs"""
        result = self.console.print("test", style="bold")
        self.assertIsNone(result)

    def test_quiet_console_status_returns_null_status(self):
        """Test that QuietConsole.status returns _NullStatus"""
        result = self.console.status("status message")
        self.assertIsInstance(result, _NullStatus)


class TestNullStatus(unittest.TestCase):
    def setUp(self):
        """Create _NullStatus instance"""
        self.status = _NullStatus()

    def test_null_status_enter_returns_none(self):
        """Test that _NullStatus.__enter__ returns None"""
        result = self.status.__enter__()
        self.assertIsNone(result)

    def test_null_status_exit_returns_false(self):
        """Test that _NullStatus.__exit__ returns False"""
        result = self.status.__exit__(None, None, None)
        self.assertFalse(result)

    def test_null_status_context_manager(self):
        """Test that _NullStatus works as context manager"""
        with self.status as s:
            self.assertIsNone(s)

    def test_null_status_context_manager_with_exception(self):
        """Test _NullStatus context manager with exception"""
        status = _NullStatus()
        try:
            with status as s:
                raise ValueError("test error")
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main()
