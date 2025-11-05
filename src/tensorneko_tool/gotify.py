import os
import socket
from . import utils
from tensorneko_util.msg import gotify as gotify_client


def register_subparser(subparsers):
    parser_gotify = subparsers.add_parser("gotify", help="Send message to Gotify server")
    parser_gotify.add_argument("message", help="The message to send", type=str)
    parser_gotify.add_argument("--title", help="The title of the message", type=str)
    parser_gotify.add_argument("--priority", help="The priority of the message", type=int, default=0)
    parser_gotify.add_argument("--url", help="The URL of the Gotify server", type=str)
    parser_gotify.add_argument("--token", help="The token of the Gotify server", type=str)
    parser_gotify.set_defaults(func=run)


def run(args):
    ok = True
    err = None
    with utils.console.status("[info]Sending Gotify message...", spinner="dots"):
        try:
            gotify_client.push(args.message, args.url, args.token, args.title, args.priority)
        except Exception as e:
            ok = False
            err = str(e)

    if ok:
        display_title = args.title or os.environ.get("GOTIFY_TITLE") or socket.gethostname()
        display_server = args.url or os.environ.get("GOTIFY_URL") or "Unknown"
        body = (
            f"Message sent successfully.\n"
            f"Title: {display_title}\n"
            f"Server: {display_server}\n"
            f"Priority: {args.priority}\n"
            f"Message: {args.message}"
        )
        utils.console.print(utils.make_panel(body, "Gotify", "green"))
        return 0
    else:
        utils.print_error(f"Failed to send Gotify message: {err}")
        return 1
