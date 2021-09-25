import os
import subprocess
from typing import Optional, Union
from shutil import copyfile

from .view import View
from ...util import tensorneko_path


class Server:
    """
    The http server for display watched components.

    Args:
        view (:class:`~tensorneko.visualization.web.view.View` | ``str``):
            The :class:`~tensorneko.visualization.web.view.View` object or the name of
            :class:`~tensorneko.visualization.web.view.View` for display.

        port (``int``, optional): The port of the http server. Default 8000.

    Attributes:
        view_name (``str``): The name of the :class:`~tensorneko.visualization.web.view.View`.
        process (:class:``subprocess.Popen``): The process of the server.

    Examples::

        # start a server
        var = tensorneko.visualization.web.Variable("x", 5)
        view = tensorneko.visualization.web.View("view1")
        view.add(var)
        server = tensorneko.visualization.web.Server(view)
        server.start()

        # stop the server
        server.stop()

    """

    def __init__(self, view: Union[View, str], port: int = 8000):
        if type(view) == View:
            self.view_name = view.name
        elif type(view) == str:
            self.view_name = view
        else:
            raise TypeError("The view type should be View or str")

        self.port = port
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """
        Start or restart the server.
        """
        # stop the old server
        if self.process is not None:
            self.stop()
        # copy files to the view directory
        source_path_index = os.path.join(tensorneko_path, "visualization", "web", "index.html")
        source_path_js = os.path.join(tensorneko_path, "visualization", "web", "main.js")
        target_path_index = os.path.join(self.view_name, "index.html")
        target_path_js = os.path.join(self.view_name, "main.js")
        if not os.path.exists(self.view_name):
            os.mkdir(self.view_name)

        copyfile(source_path_index, target_path_index)
        copyfile(source_path_js, target_path_js)

        # run python http server
        self.process = subprocess.Popen(["python", "-m", "http.server", "--directory", self.view_name, str(self.port)])
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
