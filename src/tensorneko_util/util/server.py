from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import http.server
import os
import shutil
import subprocess
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional, Union, List


class AbstractServer(ABC):

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self):
        ...

    def __enter__(self) -> AbstractServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class HttpThreadServer(AbstractServer, ABC):
    servers: List[HttpThreadServer] = []

    def __init__(self, path: str, port: int = 8000, name: str = "tensorneko server"):
        self.server_dir = path
        self.port = port
        self.name = name
        self.process: Optional[Thread] = None
        self.httpd: Optional[HTTPServer] = None

        class HTTPHandler(http.server.SimpleHTTPRequestHandler):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=path, **kwargs)

            def log_message(self, format: str, *args) -> None:
                return

        self.Handler = HTTPHandler
        self.Handler.extensions_map[".js"] = "application/javascript"
        self.Handler.extensions_map[".mjs"] = "application/javascript"
        HTTPServer.allow_reuse_address = True
        HTTPServer.allow_reuse_port = True
        self.servers.append(self)

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

    @abstractmethod
    def _prepare(self) -> None:
        pass

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
        return f"[Server] {self.name} started at http://127.0.0.1:{self.port}/."
