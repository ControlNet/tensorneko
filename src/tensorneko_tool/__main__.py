# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import argparse

from .utils import version, set_quiet
from . import utils
from .gotify import register_subparser as register_gotify
from .openai import register_subparser as register_openai

try:
    from .dep_check import register_subparser as register_dep_check
except ImportError:

    def register_dep_check(subparsers: object) -> None:
        _ = subparsers


def main():
    parser = argparse.ArgumentParser(description="TensorNeko CLI Tools")
    _ = parser.add_argument("-v", "--version", action="store_true")
    _ = parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress all stdout output"
    )
    _ = parser.add_argument(
        "-b", "--banner", action="store_true", help="Show TensorNeko title banner"
    )

    sub_parser = parser.add_subparsers(dest="sub_command")
    register_gotify(sub_parser)
    register_dep_check(sub_parser)
    register_openai(sub_parser)

    args = parser.parse_args()
    is_quiet = bool(getattr(args, "quiet", False))
    show_banner = bool(getattr(args, "banner", False))
    show_version = bool(getattr(args, "version", False))

    # Apply quiet mode globally before any output
    set_quiet(is_quiet)

    if show_banner and not is_quiet:
        utils.print_banner()

    if show_version:
        if not is_quiet and show_banner:
            # a more fancy version
            utils.console.print(
                utils.make_panel(str(version.value), "TensorNeko Version", "info")
            )
        elif not is_quiet:
            print(str(version.value))
        exit(0)

    func = getattr(args, "func", None)
    if callable(func):
        code = func(args)
        exit(code if isinstance(code, int) else 0)
    else:
        parser.print_help()
        exit(0)


if __name__ == "__main__":
    main()
