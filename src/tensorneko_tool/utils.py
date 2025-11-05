from os.path import abspath, dirname
from tensorneko_util.util.fp.monad.eval import Eval
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme


def get_tensorneko_tool_path() -> str:
    """
    Get the `tensorneko_util` library root path

    Returns:
        ``str``: The root path of `tensorneko`
    """
    return dirname(abspath(__file__))


@Eval.later
def version() -> str:
    with open(f"{get_tensorneko_tool_path()}/version.txt", "r") as file:
        return file.read().strip()


# Shared console with a simple theme for consistent CLI styling
_theme = Theme({
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "title": "bold magenta",
})
console = Console(theme=_theme)


class _NullStatus:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class QuietConsole:
    def print(self, *args, **kwargs):
        return None

    def status(self, *args, **kwargs):
        return _NullStatus()


def set_quiet(quiet: bool) -> None:
    global console
    if quiet:
        console = QuietConsole()
    else:
        console = Console(theme=_theme)


def get_banner() -> str:
    return (
        "\n"
        """████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███╗   ██╗███████╗██╗  ██╗ ██████╗ 
╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗████╗  ██║██╔════╝██║ ██╔╝██╔═══██╗
   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██╔██╗ ██║█████╗  █████╔╝ ██║   ██║
   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║╚██╗██║██╔══╝  ██╔═██╗ ██║   ██║
   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║ ╚████║███████╗██║  ██╗╚██████╔╝
   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ """
        "\n"
        "            TensorNeko CLI Tools https://github.com/ControlNet/tensorneko       \n"
    )


def print_banner() -> None:
    console.print(f"[success]{get_banner()}[/success]")


def print_info(message: str) -> None:
    console.print(f"[info]{message}[/info]")


def print_success(message: str) -> None:
    console.print(f"[success]{message}[/success]")


def print_error(message: str) -> None:
    console.print(f"[error]{message}[/error]")


def make_panel(body: str, title: str, border_style: str = "title") -> Panel:
    return Panel(body, title=f"[{border_style}]{title}[/{border_style}]", border_style=border_style)
