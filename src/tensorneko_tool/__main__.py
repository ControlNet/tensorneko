import argparse

from .utils import version, set_quiet
from . import utils
from .gotify import register_subparser as register_gotify
try:
    from .dep_check import register_subparser as register_dep_check
except ImportError:
    def register_dep_check(*args, **kwargs):
        pass

def main():
    parser = argparse.ArgumentParser(description="TensorNeko CLI Tools")
    parser.add_argument("-v", "--version", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all stdout output")
    parser.add_argument("-b", "--banner", action="store_true", help="Show TensorNeko title banner")

    sub_parser = parser.add_subparsers(dest="sub_command")
    register_gotify(sub_parser)
    register_dep_check(sub_parser)

    args = parser.parse_args()

    # Apply quiet mode globally before any output
    set_quiet(getattr(args, "quiet", False))

    if args.banner and not args.quiet:
        utils.print_banner()

    if args.version:
        if not args.quiet and args.banner:
            # a more fancy version
            utils.console.print(utils.make_panel(version.value, "TensorNeko Version", "info"))
        elif not args.quiet:
            print(version.value)
        exit(0)

    if hasattr(args, "func"):
        code = args.func(args)
        exit(code if isinstance(code, int) else 0)
    else:
        parser.print_help()
        exit(0)

if __name__ == "__main__":
    main()
