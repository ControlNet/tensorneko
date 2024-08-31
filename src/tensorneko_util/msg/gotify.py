import json
import os
import socket
import urllib.request
from typing import Optional
from urllib.error import HTTPError


def push_gotify(message: str, url: Optional[str] = None, token: Optional[str] = None, title: Optional[str] = None,
    priority: int = 0
):
    """
    Push a message to a Gotify server.

    Args:
        message (``str``): The message to be sent.
        url (``str``, optional): The URL of the Gotify server. If not provided, it will be read from the environment variable GOTIFY_URL.
        token (``str``, optional): The token for the Gotify server. If not provided, it will be read from the environment variable GOTIFY_TOKEN.
        title (``str``, optional): The title of the message. Default is GOTIFY_TITLE then the hostname of the machine.
        priority (``int``, optional): The priority of the message. Default is 0.

    Examples::

        push_gotify("This is a test message", "<URL>", "<APP_TOKEN>")
        # then the message will be sent to the Gotify server.
        # title = "<HOST_NAME>", message = "This is a test message", priority = 0

    """
    title = title or os.environ.get("GOTIFY_TITLE") or socket.gethostname()
    url = url or os.environ.get("GOTIFY_URL")
    if url is None:
        raise ValueError("URL is not provided. Input url argument or set the environment variable GOTIFY_URL.")
    token = token or os.environ.get("GOTIFY_TOKEN")
    if token is None:
        raise ValueError("Token is not provided. Input token argument or set the environment variable GOTIFY_TOKEN.")
    url = f"{url.rstrip('/')}/message?token={token}"
    data = json.dumps(
        {"title": title, "message": message, "priority": priority}
    ).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "User-Agent": "TensorNeko"
    })
    try:
        with urllib.request.urlopen(req) as res:
            pass
    except HTTPError as e:
        print(e.read().decode("utf-8"))
