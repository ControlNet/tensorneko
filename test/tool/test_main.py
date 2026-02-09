import unittest
from unittest.mock import patch, MagicMock
import sys

from tensorneko_tool.__main__ import main
from tensorneko_tool import utils


class TestMainVersion(unittest.TestCase):
    """Test main() with --version flag"""

    @patch("builtins.print")
    def test_main_version_flag_prints_version_and_exits(self, mock_print):
        """Test main() with --version flag prints version and exits 0"""
        with patch.object(sys, "argv", ["tensorneko", "--version"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn(utils.version.value, call_args)

    @patch("builtins.print")
    def test_main_version_short_flag_prints_version_and_exits(self, mock_print):
        """Test main() with -v flag prints version and exits 0"""
        with patch.object(sys, "argv", ["tensorneko", "-v"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_print.assert_called_once()


class TestMainBanner(unittest.TestCase):
    """Test main() with --banner flag"""

    @patch("tensorneko_tool.utils.print_banner")
    def test_main_banner_flag_calls_print_banner(self, mock_banner):
        """Test main() with --banner flag calls print_banner"""
        with patch.object(sys, "argv", ["tensorneko", "--banner"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_banner.assert_called_once()

    @patch("tensorneko_tool.utils.print_banner")
    def test_main_banner_short_flag_calls_print_banner(self, mock_banner):
        """Test main() with -b flag calls print_banner"""
        with patch.object(sys, "argv", ["tensorneko", "-b"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_banner.assert_called_once()


class TestMainQuiet(unittest.TestCase):
    """Test main() with --quiet flag"""

    @patch("tensorneko_tool.__main__.set_quiet")
    def test_main_quiet_flag_calls_set_quiet(self, mock_set_quiet):
        """Test main() with --quiet flag calls set_quiet(True)"""
        with patch.object(sys, "argv", ["tensorneko", "--quiet"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_set_quiet.assert_called_with(True)

    @patch("tensorneko_tool.__main__.set_quiet")
    def test_main_quiet_short_flag_calls_set_quiet(self, mock_set_quiet):
        """Test main() with -q flag calls set_quiet(True)"""
        with patch.object(sys, "argv", ["tensorneko", "-q"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_set_quiet.assert_called_with(True)


class TestMainVersionBanner(unittest.TestCase):
    """Test main() with --version and --banner flags combined"""

    @patch("tensorneko_tool.utils.make_panel")
    @patch("tensorneko_tool.utils.print_banner")
    def test_main_version_banner_fancy_output(self, mock_banner, mock_make_panel):
        """Test main() with --version --banner uses fancy panel output"""
        mock_make_panel.return_value = MagicMock()
        with patch.object(sys, "argv", ["tensorneko", "--version", "--banner"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            # make_panel should have been called with version
            mock_make_panel.assert_called_once()
            # Verify it was called with the version value
            call_args = mock_make_panel.call_args
            self.assertIn(utils.version.value, call_args[0])

    @patch("tensorneko_tool.utils.print_banner")
    def test_main_version_banner_calls_print_banner(self, mock_banner):
        """Test main() with --version --banner calls print_banner before fancy version"""
        with patch.object(sys, "argv", ["tensorneko", "--version", "--banner"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            # print_banner should be called
            self.assertTrue(mock_banner.called)


class TestMainVersionBannerQuiet(unittest.TestCase):
    """Test main() with --version, --banner, and --quiet flags combined"""

    @patch("tensorneko_tool.utils.print_banner")
    @patch("tensorneko_tool.__main__.set_quiet")
    def test_main_version_banner_quiet_no_output(self, mock_set_quiet, mock_banner):
        """Test main() with --version --banner --quiet exits 0 without printing"""
        with patch.object(sys, "argv", ["tensorneko", "--version", "--banner", "--quiet"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            # set_quiet should be called first
            mock_set_quiet.assert_called_with(True)
            # print_banner should not be called
            mock_banner.assert_not_called()


class TestMainNoArgs(unittest.TestCase):
    """Test main() with no arguments"""

    @patch("argparse.ArgumentParser.print_help")
    def test_main_no_args_prints_help(self, mock_help):
        """Test main() with no args prints help and exits 0"""
        with patch.object(sys, "argv", ["tensorneko"]):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            mock_help.assert_called_once()


class TestMainWithSubcommand(unittest.TestCase):
    """Test main() with subcommand that has func attribute"""

    def test_main_with_func_subcommand_calls_func(self):
        """Test main() calls func when subcommand has func attribute"""
        mock_func = MagicMock(return_value=0)

        with patch("tensorneko_tool.__main__.register_gotify") as mock_register_gotify:
            with patch("tensorneko_tool.__main__.register_dep_check"):
                # Mock the subparser to add a test command
                def register_test_cmd(subparsers):
                    parser = subparsers.add_parser("test")
                    parser.set_defaults(func=mock_func)

                mock_register_gotify.side_effect = register_test_cmd

                with patch.object(sys, "argv", ["tensorneko", "test"]):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 0)
                    mock_func.assert_called_once()

    def test_main_with_func_subcommand_exits_with_return_code(self):
        """Test main() exits with func's return code"""
        mock_func = MagicMock(return_value=42)

        with patch("tensorneko_tool.__main__.register_gotify") as mock_register_gotify:
            with patch("tensorneko_tool.__main__.register_dep_check"):

                def register_test_cmd(subparsers):
                    parser = subparsers.add_parser("test")
                    parser.set_defaults(func=mock_func)

                mock_register_gotify.side_effect = register_test_cmd

                with patch.object(sys, "argv", ["tensorneko", "test"]):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 42)
                    mock_func.assert_called_once()

    def test_main_with_func_subcommand_returns_non_int(self):
        """Test main() exits 0 when func returns non-int"""
        mock_func = MagicMock(return_value=None)

        with patch("tensorneko_tool.__main__.register_gotify") as mock_register_gotify:
            with patch("tensorneko_tool.__main__.register_dep_check"):

                def register_test_cmd(subparsers):
                    parser = subparsers.add_parser("test")
                    parser.set_defaults(func=mock_func)

                mock_register_gotify.side_effect = register_test_cmd

                with patch.object(sys, "argv", ["tensorneko", "test"]):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 0)
                    mock_func.assert_called_once()


if __name__ == "__main__":
    unittest.main()
