import os
import socket
import time
from typing import Optional
from . import utils
import psutil
from tensorneko_util.msg import gotify as gotify_client


def register_subparser(subparsers):
    parser_gotify = subparsers.add_parser("gotify", help="Send message to Gotify server or watch a process")
    parser_gotify.add_argument("message", nargs="?", help="Message to send (required unless using --watch)", type=str)
    parser_gotify.add_argument("--title", help="The title of the message", type=str)
    parser_gotify.add_argument("--priority", help="The priority of the message", type=int, default=0)
    parser_gotify.add_argument("--url", help="The URL of the Gotify server", type=str)
    parser_gotify.add_argument("--token", help="The token of the Gotify server", type=str)
    parser_gotify.add_argument("--watch", help="Process name or PID to watch until it exits", type=str)
    parser_gotify.add_argument("--interval", help="Polling interval in seconds for watch mode", type=float, default=5.0)
    parser_gotify.set_defaults(func=run)


def run(args):
    # Determine watch target from simplified --watch
    target_name: Optional[str] = None
    target_pid: Optional[int] = None
    if args.watch is not None:
        if args.watch.isdigit():
            target_pid = int(args.watch)
        else:
            target_name = args.watch

    if target_name is not None or target_pid is not None:
        return _run_watch(args, target_name, target_pid)
    else:
        if args.message is None:
            utils.print_error("Message is required when not using --watch")
            return 2
        return _run_send(args)


def _run_send(args):
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


def _run_watch(args, target_name: Optional[str], target_pid: Optional[int]):
    target_desc = f"pid={target_pid}" if target_pid is not None else f"name='{target_name}'"
    utils.console.print(utils.make_panel(
        f"Watching process ({target_desc}) every {args.interval}s...", "Watch then send Gotify", "yellow"
    ))

    was_seen = False  # track if process ever appeared
    try:
        while True:
            running = _is_process_running(target_name, target_pid)
            if running:
                was_seen = True
            # If it was seen before and now stopped, trigger notification
            if was_seen and not running:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        # Respect Ctrl+C: do not send unless already ended
        return 130

    # Send notification with default message if not provided
    if args.message is None:
        if target_name is not None:
            default_message = f"Process \"{target_name}\" is finished"
        else:
            default_message = f"Process PID {target_pid} is finished"
        message_to_send = default_message
    else:
        message_to_send = args.message
    return _run_send(_OverrideArgs(args, message=message_to_send))


def _is_process_running(name: Optional[str], pid: Optional[int]) -> bool:
    if pid is not None:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    if name is not None:
        name_lower = name.lower()
        for proc in psutil.process_iter(["name", "cmdline"]):
            proc_name = (proc.info.get("name") or "").lower()
            if proc_name == name_lower:
                return True
            cmd = proc.info.get("cmdline") or []
            if any(name_lower in (part or "").lower() for part in cmd):
                return True
        return False
    return False


class _OverrideArgs:
    def __init__(self, original, **overrides):
        self.__dict__.update(original.__dict__)
        for k, v in overrides.items():
            setattr(self, k, v)
