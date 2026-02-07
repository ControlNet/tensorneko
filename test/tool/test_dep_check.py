import argparse
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from packaging.requirements import Requirement
from rich.panel import Panel

import tensorneko_tool.dep_check as dep_check_module


class TestReadRequirements(unittest.TestCase):
    def test_read_requirements_parses_all_non_empty_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_file = f"{temp_dir}/requirements.txt"
            with open(requirements_file, "w") as file:
                file.write("numpy>=1.26\nrich==13.9.0\n\ntyping_extensions\n")

            requirements = dep_check_module._read_requirements(requirements_file)

            self.assertEqual(len(requirements), 3)
            self.assertEqual(requirements[0].name, "numpy")
            self.assertEqual(str(requirements[0].specifier), ">=1.26")
            self.assertEqual(requirements[1].name, "rich")
            self.assertEqual(str(requirements[1].specifier), "==13.9.0")
            self.assertEqual(requirements[2].name, "typing_extensions")


class TestGetInstalledVersion(unittest.TestCase):
    @patch("tensorneko_tool.dep_check.metadata.version", return_value="1.2.3")
    def test_get_installed_version_returns_version_when_found(self, mock_version):
        result = dep_check_module._get_installed_version("some-package")
        self.assertEqual(result, "1.2.3")
        mock_version.assert_called_once_with("some-package")

    @patch(
        "tensorneko_tool.dep_check.metadata.version",
        side_effect=dep_check_module.metadata.PackageNotFoundError,
    )
    def test_get_installed_version_returns_none_when_not_found(self, _):
        result = dep_check_module._get_installed_version("missing-package")
        self.assertIsNone(result)

    @patch(
        "tensorneko_tool.dep_check.metadata.version",
        side_effect=dep_check_module.InvalidVersion("bad-version"),
    )
    @patch("tensorneko_tool.dep_check.utils.console")
    def test_get_installed_version_prints_error_and_returns_none_for_invalid_version(
        self, mock_console, _
    ):
        result = dep_check_module._get_installed_version("bad-package")
        self.assertIsNone(result)
        mock_console.print.assert_called_once_with(
            "[bold red]Invalid version for package bad-package[/bold red]"
        )


class TestCheckRequirements(unittest.TestCase):
    @patch(
        "tensorneko_tool.dep_check._get_installed_version",
        side_effect=["2.1.0", "13.0.1"],
    )
    def test_check_requirements_returns_empty_lists_when_all_satisfied(self, _):
        requirements = [Requirement("numpy>=1.0"), Requirement("rich==13.0.1")]

        missing, mismatches = dep_check_module._check_requirements(requirements)

        self.assertEqual(missing, [])
        self.assertEqual(mismatches, [])

    @patch("tensorneko_tool.dep_check._get_installed_version", side_effect=[None])
    def test_check_requirements_collects_missing_packages(self, _):
        requirement = Requirement("not-installed>=1.0")

        missing, mismatches = dep_check_module._check_requirements([requirement])

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0][0], "not-installed")
        self.assertEqual(str(missing[0][1]), ">=1.0")
        self.assertEqual(mismatches, [])

    @patch("tensorneko_tool.dep_check._get_installed_version", side_effect=["1.0.0"])
    def test_check_requirements_collects_version_mismatches(self, _):
        requirement = Requirement("mypkg>=2.0")

        missing, mismatches = dep_check_module._check_requirements([requirement])

        self.assertEqual(missing, [])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0][0], "mypkg")
        self.assertEqual(mismatches[0][1], "1.0.0")
        self.assertEqual(str(mismatches[0][2]), ">=2.0")

    @patch("tensorneko_tool.dep_check._get_installed_version", side_effect=["1.0.0"])
    def test_check_requirements_allows_any_version_when_no_specifier(self, _):
        requirement = Requirement("mypkg")

        missing, mismatches = dep_check_module._check_requirements([requirement])

        self.assertEqual(missing, [])
        self.assertEqual(mismatches, [])


class TestDisplayResults(unittest.TestCase):
    @patch("tensorneko_tool.dep_check.utils.console")
    def test_display_results_prints_success_when_no_issues(self, mock_console):
        dep_check_module._display_results([], [])

        mock_console.print.assert_called_once_with(
            "[success]All dependencies are satisfied.[/success]"
        )

    @patch("tensorneko_tool.dep_check.utils.console")
    def test_display_results_prints_missing_packages_panel(self, mock_console):
        missing_packages = [("missinglib", Requirement("missinglib>=1.0").specifier)]

        dep_check_module._display_results(missing_packages, [])

        printed_panel = mock_console.print.call_args[0][0]
        self.assertIsInstance(printed_panel, Panel)
        self.assertIn("Missing Packages (1)", str(printed_panel.title))
        self.assertEqual(printed_panel.border_style, "red")

    @patch("tensorneko_tool.dep_check.utils.console")
    def test_display_results_prints_version_mismatches_panel(self, mock_console):
        mismatches = [("somelib", "1.0.0", Requirement("somelib>=2.0").specifier)]

        dep_check_module._display_results([], mismatches)

        printed_panel = mock_console.print.call_args[0][0]
        self.assertIsInstance(printed_panel, Panel)
        self.assertIn("Version Mismatches (1)", str(printed_panel.title))
        self.assertEqual(printed_panel.border_style, "yellow")

    @patch("tensorneko_tool.dep_check.utils.console")
    def test_display_results_prints_both_panels_when_both_issue_types_exist(
        self, mock_console
    ):
        missing_packages = [("missinglib", Requirement("missinglib>=1.0").specifier)]
        mismatches = [("somelib", "1.0.0", Requirement("somelib>=2.0").specifier)]

        dep_check_module._display_results(missing_packages, mismatches)

        self.assertEqual(mock_console.print.call_count, 2)
        first_panel = mock_console.print.call_args_list[0][0][0]
        second_panel = mock_console.print.call_args_list[1][0][0]
        self.assertIn("Missing Packages (1)", str(first_panel.title))
        self.assertIn("Version Mismatches (1)", str(second_panel.title))


class TestOverwriteRequirements(unittest.TestCase):
    def test_overwrite_requirements_rewrites_mismatched_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_file = f"{temp_dir}/requirements.txt"
            with open(requirements_file, "w") as file:
                file.write("numpy>=1.0\nrich==13.0.0\ntqdm\n")

            requirements = dep_check_module._read_requirements(requirements_file)
            mismatched = [
                ("numpy", "1.26.4", Requirement("numpy>=1.0").specifier),
                ("rich", "13.9.4", Requirement("rich==13.0.0").specifier),
            ]

            dep_check_module._overwrite_requirements(
                requirements_file, requirements, mismatched
            )

            with open(requirements_file, "r") as file:
                content = file.read().splitlines()

            self.assertEqual(content[0], "numpy==1.26.4")
            self.assertEqual(content[1], "rich==13.9.4")
            self.assertEqual(content[2], "tqdm")


class TestDepCheck(unittest.TestCase):
    @patch("tensorneko_tool.dep_check._overwrite_requirements")
    @patch("tensorneko_tool.dep_check._display_results")
    @patch("tensorneko_tool.dep_check._check_requirements", return_value=([], []))
    @patch(
        "tensorneko_tool.dep_check._read_requirements",
        return_value=[Requirement("numpy>=1.0")],
    )
    @patch("tensorneko_tool.dep_check.utils.console")
    def test_dep_check_runs_without_overwrite_when_no_mismatch(
        self, mock_console, mock_read, mock_check, mock_display, mock_overwrite
    ):
        dep_check_module.dep_check("requirements.txt", overwrite=False)

        mock_read.assert_called_once_with("requirements.txt")
        mock_check.assert_called_once()
        mock_display.assert_called_once_with([], [])
        mock_overwrite.assert_not_called()
        mock_console.status.assert_called_once_with(
            "[info]Checking dependencies...", spinner="dots"
        )

    @patch("tensorneko_tool.dep_check._overwrite_requirements")
    @patch("tensorneko_tool.dep_check._display_results")
    @patch(
        "tensorneko_tool.dep_check._check_requirements",
        return_value=([], [("numpy", "1.26.4", Requirement("numpy>=2.0").specifier)]),
    )
    @patch(
        "tensorneko_tool.dep_check._read_requirements",
        return_value=[Requirement("numpy>=2.0")],
    )
    @patch("tensorneko_tool.dep_check.utils.console")
    def test_dep_check_overwrites_when_enabled_and_mismatch_exists(
        self, mock_console, mock_read, _, mock_display, mock_overwrite
    ):
        dep_check_module.dep_check("requirements.txt", overwrite=True)

        mock_read.assert_called_once_with("requirements.txt")
        mock_display.assert_called_once()
        mock_overwrite.assert_called_once_with(
            "requirements.txt",
            [Requirement("numpy>=2.0")],
            [("numpy", "1.26.4", Requirement("numpy>=2.0").specifier)],
        )
        mock_console.print.assert_called_with(
            "[success]Requirements file requirements.txt has been updated with installed versions.[/success]"
        )


class TestRegisterSubparser(unittest.TestCase):
    def test_register_subparser_sets_defaults_and_arguments(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        dep_check_module.register_subparser(subparsers)
        args = parser.parse_args(["dep_check"])

        self.assertEqual(args.requirements, "requirements.txt")
        self.assertFalse(args.overwrite)
        self.assertIs(args.func, dep_check_module.run)

    def test_register_subparser_parses_custom_arguments(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        dep_check_module.register_subparser(subparsers)
        args = parser.parse_args(["dep_check", "-r", "custom.txt", "--overwrite"])

        self.assertEqual(args.requirements, "custom.txt")
        self.assertTrue(args.overwrite)


class TestRun(unittest.TestCase):
    @patch("tensorneko_tool.dep_check.utils.console")
    def test_run_returns_one_when_dep_check_disabled(self, mock_console):
        args = Namespace(requirements="requirements.txt", overwrite=False)

        with patch.object(dep_check_module, "disable_dep_check", True):
            result = dep_check_module.run(args)

        self.assertEqual(result, 1)
        mock_console.print.assert_called_once_with(
            "[error]dep_check is only available in Python 3.8 and above.[/error]"
        )

    @patch("tensorneko_tool.dep_check.dep_check")
    def test_run_calls_dep_check_and_returns_zero_when_enabled(self, mock_dep_check):
        args = Namespace(requirements="requirements.txt", overwrite=True)

        with patch.object(dep_check_module, "disable_dep_check", False):
            result = dep_check_module.run(args)

        self.assertEqual(result, 0)
        mock_dep_check.assert_called_once_with("requirements.txt", True)


if __name__ == "__main__":
    unittest.main()
