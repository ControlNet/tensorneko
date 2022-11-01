from __future__ import annotations

import json
import os
import shutil

from ...util import tensorneko_util_path
from ...util.server import HttpThreadServer


class Server(HttpThreadServer):

    def __init__(self, path: str, port: int = 8000):
        super().__init__(path, port, "Tensorneko Image Browser")

    def _prepare(self) -> None:
        # write a metadata json file list oll images in that directory
        metadata = []
        for file in os.listdir(self.server_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                metadata.append(file)
        with open(os.path.join(self.server_dir, ".metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=True)

        # copy index.html to target directory
        shutil.copyfile(
            os.path.join(tensorneko_util_path, "visualization", "image_browser", "web", "index.html"),
            os.path.join(self.server_dir, "index.html")
        )

    def stop(self) -> None:
        super().stop()
        # remove the metadata file
        os.remove(os.path.join(self.server_dir, ".metadata.json"))
        # remove the index.html file
        os.remove(os.path.join(self.server_dir, "index.html"))
