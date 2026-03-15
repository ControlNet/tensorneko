import subprocess
from typing import Optional

from ..util.server import AbstractServer


class Server(AbstractServer):
    def __init__(self, logdir: str = "logs", port: int = 6006):
        self.logdir = logdir
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self.process is not None:
            self.stop()

        self.process = subprocess.Popen(
            ["tensorboard", "--logdir", self.logdir, "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print(f'Server started at port {self.port}, logdir "{self.logdir}".')

    def start_blocking(self) -> None:
        if self.process is not None:
            self.stop()

        subprocess.run(
            ["tensorboard", "--logdir", self.logdir, "--port", str(self.port)]
        )
        print(f'Server started at port {self.port}, logdir "{self.logdir}".')

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            print("Server stopped")
            self.process = None
        else:
            print("Server is stopped.")
