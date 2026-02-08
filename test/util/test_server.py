import os
import tempfile
import unittest
import urllib.request

from tensorneko_util.util.server import AbstractServer, HttpThreadServer


PORT_BASE = 18750


class MinimalServer(HttpThreadServer):
    """Minimal concrete subclass for testing HttpThreadServer."""

    def _prepare(self) -> None:
        # Write a simple index.html into the server directory
        index_path = os.path.join(self.server_dir, "index.html")
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                f.write("<html><body>OK</body></html>")


class TestHttpThreadServerLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Clear class-level servers list before each test
        HttpThreadServer.servers.clear()

    def tearDown(self):
        # Stop all servers and clear tracking list
        for server in list(HttpThreadServer.servers):
            try:
                server.stop()
            except Exception:
                pass
        HttpThreadServer.servers.clear()
        # Clean up temp dir
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_start_and_stop(self):
        """Test basic start/stop lifecycle."""
        port = PORT_BASE + 1
        server = MinimalServer(self.tmpdir, port=port, name="test-start-stop")
        server.start()
        self.assertIsNotNone(server.process)
        self.assertIsNotNone(server.httpd)

        # Verify the server is actually serving
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        content = resp.read().decode()
        self.assertIn("OK", content)
        resp.close()

        server.stop()
        self.assertIsNone(server.process)

    def test_context_manager(self):
        """Test __enter__ / __exit__ context manager protocol."""
        port = PORT_BASE + 2
        server = MinimalServer(self.tmpdir, port=port, name="test-ctx")
        with server as s:
            self.assertIs(s, server)
            self.assertIsNotNone(server.process)
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            content = resp.read().decode()
            self.assertIn("OK", content)
            resp.close()
        # After exiting context, server should be stopped
        self.assertIsNone(server.process)

    def test_stop_all(self):
        """Test the stop_all class method stops all tracked servers."""
        port1 = PORT_BASE + 3
        port2 = PORT_BASE + 4
        s1 = MinimalServer(self.tmpdir, port=port1, name="test-all-1")
        s2 = MinimalServer(self.tmpdir, port=port2, name="test-all-2")
        s1.start()
        s2.start()
        self.assertIsNotNone(s1.process)
        self.assertIsNotNone(s2.process)

        HttpThreadServer.stop_all()
        self.assertIsNone(s1.process)
        self.assertIsNone(s2.process)

    def test_servers_list_tracking(self):
        """Test that servers are tracked in the class-level list."""
        self.assertEqual(len(HttpThreadServer.servers), 0)
        port = PORT_BASE + 5
        s1 = MinimalServer(self.tmpdir, port=port, name="test-tracking")
        self.assertEqual(len(HttpThreadServer.servers), 1)
        self.assertIs(HttpThreadServer.servers[0], s1)

    def test_double_stop_is_safe(self):
        """Test that calling stop() twice doesn't raise."""
        port = PORT_BASE + 6
        server = MinimalServer(self.tmpdir, port=port, name="test-double-stop")
        server.start()
        server.stop()
        self.assertIsNone(server.process)
        # Second stop should not raise
        server.stop()
        self.assertIsNone(server.process)

    def test_restart(self):
        """Test that calling start() when already running restarts the server."""
        port1 = PORT_BASE + 7
        port2 = PORT_BASE + 10
        server = MinimalServer(self.tmpdir, port=port1, name="test-restart")
        server.start()
        old_process = server.process
        server.port = port2
        server.start()
        self.assertIsNotNone(server.process)
        self.assertIsNot(server.process, old_process)
        server.stop()

    def test_server_start_string(self):
        """Test the server_start_string property."""
        port = PORT_BASE + 8
        server = MinimalServer(self.tmpdir, port=port, name="my-test-server")
        expected = f"[Server] my-test-server started at http://127.0.0.1:{port}/."
        self.assertEqual(server.server_start_string, expected)

    def test_init_attributes(self):
        """Test that __init__ sets attributes correctly."""
        port = PORT_BASE + 9
        server = MinimalServer(self.tmpdir, port=port, name="attr-test")
        self.assertEqual(server.server_dir, self.tmpdir)
        self.assertEqual(server.port, port)
        self.assertEqual(server.name, "attr-test")
        self.assertIsNone(server.process)
        self.assertIsNone(server.httpd)
        self.assertIsNotNone(server.Handler)


if __name__ == "__main__":
    unittest.main()
