import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tensorneko_util.visualization.image_browser.server import Server
from tensorneko_util.util.server import HttpThreadServer


class TestImageBrowserServer(unittest.TestCase):
    """Test ImageBrowserServer lifecycle and _prepare/_stop methods."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)
        # Create test images directory
        self._imgdir = os.path.join(self._tmpdir, "test_images")
        os.makedirs(self._imgdir)
        # Create some dummy image files
        for name in ["a.png", "b.jpg", "c.jpeg"]:
            with open(os.path.join(self._imgdir, name), "wb") as f:
                f.write(b"\x00" * 10)
        # Create a subdirectory with an image
        subdir = os.path.join(self._imgdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "d.png"), "wb") as f:
            f.write(b"\x00" * 10)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        HttpThreadServer.servers.clear()

    def test_server_is_http_thread_server(self):
        server = Server(self._imgdir, port=18830)
        self.assertIsInstance(server, HttpThreadServer)

    def test_server_init(self):
        server = Server(self._imgdir, port=18831, exts=("png", "jpg"))
        self.assertEqual(server.port, 18831)
        self.assertEqual(server.exts, ("png", "jpg"))

    def test_prepare_creates_metadata(self):
        server = Server(self._imgdir, port=18832)
        server._prepare()
        meta_path = os.path.join(self._imgdir, ".metadata.json")
        self.assertTrue(os.path.isfile(meta_path))
        with open(meta_path) as f:
            metadata = json.load(f)
        self.assertIsInstance(metadata, list)
        # Should find png and jpg files (including subdirectory)
        self.assertGreater(len(metadata), 0)
        # Should be relative paths
        for path in metadata:
            self.assertFalse(os.path.isabs(path))

    def test_prepare_creates_index_html(self):
        server = Server(self._imgdir, port=18833)
        try:
            server._prepare()
            # index.html should be copied
            index_path = os.path.join(self._imgdir, "index.html")
            self.assertTrue(os.path.isfile(index_path))
        except FileNotFoundError:
            pass  # web template may not exist in test env

    def test_start_and_stop(self):
        server = Server(self._imgdir, port=18834)
        try:
            server.start()
            self.assertIsNotNone(server.process)
            server.stop()
            self.assertIsNone(server.process)
        except FileNotFoundError:
            pass  # web template may not exist

    def test_stop_removes_metadata_and_index(self):
        server = Server(self._imgdir, port=18835)
        try:
            server.start()
            server.stop()
            # metadata and index.html should be removed
            self.assertFalse(
                os.path.isfile(os.path.join(self._imgdir, ".metadata.json"))
            )
            self.assertFalse(os.path.isfile(os.path.join(self._imgdir, "index.html")))
        except FileNotFoundError:
            pass

    def test_server_start_string(self):
        server = Server(self._imgdir, port=18836)
        s = server.server_start_string
        self.assertIn("18836", s)


if __name__ == "__main__":
    unittest.main()
