from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Union, List

from .view import View
from ...util import tensorneko_path


class Server:
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

    def __init__(self, view: Union[View, str], port: int = 8000):
        if type(view) == View:
            self.view_name = view.name
        elif type(view) == str:
            self.view_name = view
        else:
            raise TypeError("The view type should be View or str")

        self.port = port
        self.process: Optional[subprocess.Popen] = None
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
        source_path = os.path.join(tensorneko_path, "visualization", "watcher", "web", "dist")
        target_path = os.path.join(self.view_name)
        if not os.path.exists(self.view_name):
            os.mkdir(self.view_name)

        target_path_css = os.path.join(self.view_name, "css")
        if os.path.exists(target_path_css):
            shutil.rmtree(target_path_css)

        target_path_js = os.path.join(self.view_name, "js")
        if os.path.exists(target_path_js):
            shutil.rmtree(target_path_js)

        target_path_html = os.path.join(self.view_name, "index.html")
        if os.path.exists(target_path_html):
            os.remove(target_path_html)

        shutil.copy(os.path.join(source_path, "index.html"), target_path_html)
        shutil.copytree(os.path.join(source_path, "css"), target_path_css)
        shutil.copytree(os.path.join(source_path, "js"), target_path_js)

    def _run(self) -> None:
        self.process = subprocess.Popen(["python", "-m", "http.server", "--directory", self.view_name, str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print(f"Server started at port {self.port}, view \"{self.view_name}\".")

    def _run_blocking(self):
        self.process = subprocess.run(["python", "-m", "http.server", "--directory", self.view_name, str(self.port)])
        print(f"Server started at port {self.port}, view \"{self.view_name}\".")

    def stop(self) -> None:
        """
        Stop the server.
        """
        if self.process is not None:
            self.process.terminate()
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

    def __enter__(self) -> Server:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
