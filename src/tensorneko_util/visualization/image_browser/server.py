from __future__ import annotations

import json
import os
import shutil
import glob
from typing import Collection

from ...util import tensorneko_util_path
from ...util.server import HttpThreadServer


class Server(HttpThreadServer):

    def __init__(self, path: str, port: int = 8000, exts: Collection[str] = ("png", "jpg", "jpeg")):
        super().__init__(path, port, "Tensorneko Image Browser")
        self.exts = exts

    def _prepare(self) -> None:
        # write a metadata json file list oll images in that directory
        metadata = []
        for ext in self.exts:
            metadata.extend(glob.glob(os.path.join(self.server_dir, "**", f"*.{ext}"), recursive=True))

        metadata.sort()
        metadata = [os.path.relpath(x, self.server_dir) for x in metadata]

        with open(os.path.join(self.server_dir, ".metadata.json"), "w") as f:
            json.dump(sorted(metadata), f, indent=4, ensure_ascii=True)

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
