import subprocess

from tensorneko_util.util import AbstractServer


class Server(AbstractServer):
    def __init__(self, logdir: str = "logs", port: int = 6006):
        self.logdir = logdir
        self.port = port
        self.process = None

    def start(self) -> None:
        # stop the old server
        if self.process is not None:
            self.stop()

        # run new server
        self.process = subprocess.Popen(["tensorboard", "--logdir", self.logdir, "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print(f"Server started at port {self.port}, logdir \"{self.logdir}\".")

    def start_blocking(self) -> None:
        # stop the old server
        if self.process is not None:
            self.stop()

        # run new server
        self.process = subprocess.run(["tensorboard", "--logdir", self.logdir, "--port", str(self.port)])
        print(f"Server started at port {self.port}, logdir \"{self.logdir}\".")

    def stop(self):
        """
        Stop the server.
        """
        if self.process is not None:
            self.process.terminate()
            print("Server stopped")
            self.process = None
        else:
            print("Server is stopped.")
