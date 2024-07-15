from os.path import abspath, dirname
from tensorneko_util.util.fp.monad.eval import Eval


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
