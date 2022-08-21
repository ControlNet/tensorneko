from __future__ import annotations

import http.server
import os
import shutil
import subprocess
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional, Union, List

from .view import View
from ...util import tensorneko_util_path, AbstractServer


class Server(AbstractServer):
    """
    The http server for display watched components.

    Args:
        view (:class:`~tensorneko.visualization.web.view.View` | ``str``):
            The :class:`~tensorneko.visualization.watcher.view.View` object or the name of
            :class:`~tensorneko.visualization.watcher.view.View` for display.

        port (``int``, optional): The port of the http server. Default 8000.

    Attributes:
        view_name (``str``): The name of the :class:`~tensorneko.visualization.watcher.view.View`.
        process (:class:``subprocess.Popen``): The process of the server.
        save_dir (``str``): The directory to save the web files.

    Examples::

        # start a server
        var = tensorneko.visualization.watcher.Variable("x", 5)
        view = tensorneko.visualization.watcher.View("view1")
        view.add(var)
        server = tensorneko.visualization.watcher.Server(view)
        server.start()

        # stop the server
        server.stop()

        # start the server on blocking
        server.start_blocking()

    """
    servers: List[Server] = []

    def __init__(self, view: Union[View, str], port: int = 8000, save_dir: str = None):
        if type(view) == View:
            self.view_name = view.name
        elif type(view) == str:
            self.view_name = view
        else:
            raise TypeError("The view type should be View or str")

        self.save_dir = save_dir or os.path.join("watcher", self.view_name)
        self.port = port
        self.process: Optional[Thread] = None
        self.httpd: Optional[HTTPServer] = None
        server_dir = os.path.join("watcher", self.view_name)

        class HTTPHandler(http.server.SimpleHTTPRequestHandler):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=server_dir, **kwargs)

            def log_message(self, format: str, *args) -> None:
                return

        self.Handler = HTTPHandler
        self.Handler.extensions_map[".js"] = "application/javascript"
        self.Handler.extensions_map[".mjs"] = "application/javascript"
        http.server.HTTPServer.allow_reuse_address = True
        http.server.HTTPServer.allow_reuse_port = True
        Server.servers.append(self)

    def start(self) -> None:
        """
        Start or restart the server.
        """
        # stop the old server
        if self.process is not None:
            self.stop()
        # copy files to the view directory
        self._prepare()
        # run python http server
        self._run()

    def start_blocking(self) -> None:
        """
        Start or restart the server on blocking.
        """
        # stop the old server
        if self.process is not None:
            self.stop()
        # copy files to the view directory
        self._prepare()
        # run python http server
        self._run_blocking()

    def _prepare(self) -> None:
        source_path = os.path.join(tensorneko_util_path, "visualization", "watcher", "web", "dist")
        target_path = self.save_dir
        if not os.path.exists(target_path):
            Path(target_path).mkdir(parents=True, exist_ok=True)

        target_path_assets = os.path.join(target_path, "assets")
        if os.path.exists(target_path_assets):
            shutil.rmtree(target_path_assets)

        target_path_html = os.path.join(target_path, "index.html")
        if os.path.exists(target_path_html):
            os.remove(target_path_html)

        shutil.copy(os.path.join(source_path, "index.html"), target_path_html)
        shutil.copytree(os.path.join(source_path, "assets"), target_path_assets)

    def _run(self) -> None:
        self.httpd = http.server.HTTPServer(("", self.port), self.Handler)
        self.process = Thread(target=self.httpd.serve_forever)
        self.process.start()
        print(self.server_start_string)

    def _run_blocking(self):
        with http.server.HTTPServer(("", self.port), self.Handler) as httpd:
            print(self.server_start_string)
            httpd.serve_forever()

    def stop(self) -> None:
        """
        Stop the server.
        """
        if self.process is not None:
            self.httpd.shutdown()
            self.process.join()
            print("Server stopped")
            self.process = None
        else:
            print("Server is stopped.")

    @classmethod
    def stop_all(cls) -> None:
        """
        Stop all servers.
        """
        for server in cls.servers:
            server.stop()

    @property
    def server_start_string(self) -> str:
        return f"Server started at http://127.0.0.1:{self.port}/, view \"{self.view_name}\"."
