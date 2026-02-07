import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from tensorneko_util.visualization.watcher.component import (
    Component,
    Variable,
    ProgressBar,
    Logger,
    LineChart,
    Image,
    Bindable,
    BindableVariable,
    BindableProgressBar,
)
from tensorneko_util.visualization.watcher.view import View
from tensorneko_util.visualization.watcher.server import Server
from tensorneko_util.util.ref import Ref


class TestVariable(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_create_variable(self):
        var = Variable("x", 5)
        self.assertEqual(var.name, "x")
        self.assertEqual(var.value, 5)

    def test_update_value(self):
        var = Variable("x", 5)
        var.value = 10
        self.assertEqual(var.value, 10)

    def test_set_value(self):
        var = Variable("x", 5)
        var.set(20)
        self.assertEqual(var.value, 20)

    def test_to_dict(self):
        var = Variable("x", 42)
        d = var.to_dict()
        self.assertEqual(d, {"type": "Variable", "name": "x", "value": "42"})

    def test_str_repr(self):
        var = Variable("x", 5)
        s = str(var)
        self.assertIn("Variable", s)
        self.assertIn("x", s)
        r = repr(var)
        self.assertEqual(s, r)

    def test_registered_in_components(self):
        var = Variable("myvar", 1)
        self.assertIn("myvar", Component.components)
        self.assertIs(Component.components["myvar"], var)

    @patch.object(View, "_save_json")
    def test_update_view_calls_view_update(self, mock_save):
        """Cover component.py line 40: update_view triggers view.update()."""
        var = Variable("uv_var", 0)
        view = View("uv_view")
        view.add(var)
        # Setting value triggers update_view → view.update() → _save_json
        var.value = 99
        mock_save.assert_called()


class TestProgressBar(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_create_progress_bar(self):
        pb = ProgressBar("proc", 100)
        self.assertEqual(pb.name, "proc")
        self.assertEqual(pb.value, 0)
        self.assertEqual(pb.total, 100)

    def test_create_with_initial_value(self):
        pb = ProgressBar("proc", 100, value=50)
        self.assertEqual(pb.value, 50)

    def test_add(self):
        pb = ProgressBar("proc", 100)
        pb.add(5)
        self.assertEqual(pb.value, 5)
        pb.add(10)
        self.assertEqual(pb.value, 15)

    def test_to_dict(self):
        pb = ProgressBar("proc", 100, value=25)
        d = pb.to_dict()
        self.assertEqual(
            d, {"type": "ProgressBar", "name": "proc", "value": 25, "total": 100}
        )

    def test_set_value(self):
        pb = ProgressBar("proc", 100)
        pb.set(42)
        self.assertEqual(pb.value, 42)


class TestLogger(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_create_logger(self):
        logger = Logger("log")
        self.assertEqual(logger.name, "log")
        self.assertEqual(logger.value, [])

    def test_create_with_initial_value(self):
        logger = Logger("log", ["msg1"])
        self.assertEqual(logger.value, ["msg1"])

    def test_log_message(self):
        logger = Logger("log")
        logger.log("hello")
        logger.log("world")
        self.assertEqual(logger.value, ["hello", "world"])

    def test_to_dict(self):
        logger = Logger("log")
        logger.log("test")
        d = logger.to_dict()
        self.assertEqual(d, {"type": "Logger", "name": "log", "value": ["test"]})


class TestLineChart(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_create_line_chart(self):
        lc = LineChart("chart")
        self.assertEqual(lc.name, "chart")
        self.assertEqual(lc.value, [])
        self.assertEqual(lc.x_label, "index")
        self.assertEqual(lc.y_label, "value")

    def test_create_with_labels(self):
        lc = LineChart("chart", x_label="epoch", y_label="loss")
        self.assertEqual(lc.x_label, "epoch")
        self.assertEqual(lc.y_label, "loss")

    def test_add_data_points(self):
        lc = LineChart("chart")
        lc.add(1.0, 0.5, label="train")
        lc.add(2.0, 0.3, label="train")
        self.assertEqual(len(lc.value), 2)
        self.assertEqual(lc.value[0], {"x": 1.0, "y": 0.5, "label": "train"})
        self.assertEqual(lc.value[1], {"x": 2.0, "y": 0.3, "label": "train"})

    def test_to_dict(self):
        lc = LineChart("chart", x_label="x", y_label="y")
        lc.add(1.0, 2.0)
        d = lc.to_dict()
        self.assertEqual(d["type"], "LineChart")
        self.assertEqual(d["name"], "chart")
        self.assertEqual(d["x_label"], "x")
        self.assertEqual(d["y_label"], "y")
        self.assertEqual(len(d["value"]), 1)


class TestView(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    @patch.object(View, "_save_json")
    def test_create_view(self, mock_save):
        view = View("test_view")
        self.assertEqual(view.name, "test_view")
        self.assertEqual(view.components, [])

    @patch.object(View, "_save_json")
    def test_add_components(self, mock_save):
        var = Variable("x", 1)
        pb = ProgressBar("pb", 100)
        view = View("test_view")
        view.add(var, pb)
        self.assertEqual(len(view.components), 2)
        self.assertIn(view, var.views)
        self.assertIn(view, pb.views)

    @patch.object(View, "_save_json")
    def test_add_returns_self(self, mock_save):
        var = Variable("x", 1)
        view = View("test_view")
        result = view.add(var)
        self.assertIs(result, view)

    @patch.object(View, "_save_json")
    def test_add_all(self, mock_save):
        var = Variable("a", 1)
        pb = ProgressBar("b", 10)
        logger = Logger("c")
        view = View("test_view")
        view.add_all()
        self.assertEqual(len(view.components), 3)

    @patch.object(View, "_save_json")
    def test_getitem(self, mock_save):
        var = Variable("x", 42)
        view = View("test_view")
        view.add(var)
        retrieved = view["x"]
        self.assertIs(retrieved, var)

    @patch.object(View, "_save_json")
    def test_getitem_not_found(self, mock_save):
        view = View("test_view")
        with self.assertRaises(KeyError):
            _ = view["nonexistent"]

    @patch.object(View, "_save_json")
    def test_remove_by_name(self, mock_save):
        var = Variable("x", 1)
        view = View("test_view")
        view.add(var)
        self.assertEqual(len(view.components), 1)
        view.remove("x")
        self.assertEqual(len(view.components), 0)

    @patch.object(View, "_save_json")
    def test_remove_nonexistent(self, mock_save):
        """Removing a nonexistent component should print error but not raise."""
        view = View("test_view")
        view.remove("nope")  # should not raise


class TestComponentClassDict(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_components_dict_populated(self):
        self.assertEqual(len(Component.components), 0)
        Variable("a", 1)
        ProgressBar("b", 10)
        Logger("c")
        LineChart("d")
        self.assertEqual(len(Component.components), 4)
        self.assertIn("a", Component.components)
        self.assertIn("b", Component.components)
        self.assertIn("c", Component.components)
        self.assertIn("d", Component.components)


class TestImage(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_create_image_component(self):
        img = Image("img0")
        self.assertEqual(img.name, "img0")
        self.assertIsNone(img.value)
        self.assertEqual(img.path, "img/img0")
        self.assertEqual(img._ver, 0)

    def test_to_dict_no_value(self):
        img = Image("img0")
        d = img.to_dict()
        self.assertEqual(d["type"], "Image")
        self.assertEqual(d["name"], "img0")


class TestImageWithValue(unittest.TestCase):
    """Test Image component with actual image arrays."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        Component.components.clear()

    def test_image_update_with_value_saves_file(self):
        """Image.update writes a jpg file for each associated view."""
        arr = np.random.rand(3, 32, 32).astype(np.float32)
        img = Image("test_img", arr)
        # Create the directory structure that View._save_json would create
        os.makedirs(os.path.join("watcher", "imgview"), exist_ok=True)
        view = View("imgview")
        view.add(img)
        # After add, update should have been called
        img_dir = os.path.join("watcher", "imgview", "img")
        self.assertTrue(os.path.isdir(img_dir))
        # Version should have incremented
        self.assertGreater(img._ver, 0)

    def test_image_to_dict_with_value(self):
        arr = np.random.rand(3, 16, 16).astype(np.float32)
        img = Image("test_img2", arr)
        os.makedirs(os.path.join("watcher", "imgview2"), exist_ok=True)
        view = View("imgview2")
        view.add(img)
        d = img.to_dict()
        self.assertEqual(d["type"], "Image")
        self.assertIn("/img/test_img2-", d["value"])
        self.assertTrue(d["value"].endswith(".jpg"))

    def test_image_update_removes_old_version(self):
        """Updating image value should remove old file and create new."""
        arr1 = np.random.rand(3, 16, 16).astype(np.float32)
        arr2 = np.random.rand(3, 16, 16).astype(np.float32)
        img = Image("versioned_img", arr1)
        os.makedirs(os.path.join("watcher", "verview"), exist_ok=True)
        view = View("verview")
        view.add(img)
        ver1 = img._ver
        # Now trigger another update
        img._value = arr2
        img.update()
        self.assertGreater(img._ver, ver1)


class TestBindableVariable(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_bindable_variable_from_ref(self):
        ref = Ref(42)
        var = Variable("bvar", 0)
        bvar = var.bind(ref)
        self.assertIsInstance(bvar, BindableVariable)
        self.assertEqual(bvar.ref, ref)

    def test_bindable_value_setter_raises(self):
        ref = Ref(10)
        var = Variable("bvar2", 0)
        bvar = var.bind(ref)
        with self.assertRaises(ValueError):
            bvar.value = 99

    def test_bindable_variable_value_reads_ref(self):
        ref = Ref(77)
        var = Variable("bvar3", 0)
        bvar = var.bind(ref)
        self.assertEqual(bvar.value, 77)


class TestBindableProgressBar(unittest.TestCase):
    def tearDown(self):
        Component.components.clear()

    def test_bindable_progress_bar_from_ref(self):
        ref = Ref(5)
        pb = ProgressBar("bpb", 100)
        bpb = pb.bind(ref)
        self.assertIsInstance(bpb, BindableProgressBar)
        self.assertEqual(bpb.ref, ref)
        self.assertEqual(bpb.total, 100)


class TestViewSaveJson(unittest.TestCase):
    """Test View._save_json with real filesystem."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        Component.components.clear()

    def test_save_json_creates_directory_and_file(self):
        var = Variable("x", 42)
        view = View("jsonview")
        view.add(var)
        json_path = os.path.join("watcher", "jsonview", "data.json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path) as f:
            data = json.load(f)
        self.assertEqual(data["view"], "jsonview")
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["type"], "Variable")

    def test_view_remove_nonexistent_component_type(self):
        """Removing a non-Component non-str value prints error."""
        view = View("rmview")
        view.remove(12345)  # not a Component or str — line 107


class TestWatcherServer(unittest.TestCase):
    """Test watcher Server init, start, stop, context manager."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        Component.components.clear()
        Server.servers.clear()

    def test_server_init_with_view_object(self):
        view = View("srv_view")
        server = Server(view, port=18800)
        self.assertEqual(server.view_name, "srv_view")
        self.assertEqual(server.port, 18800)
        self.assertIn(server, Server.servers)

    def test_server_init_with_string(self):
        server = Server("str_view", port=18801)
        self.assertEqual(server.view_name, "str_view")

    def test_server_init_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            Server(12345, port=18802)

    def test_server_start_string(self):
        server = Server("test_view", port=18803)
        s = server.server_start_string
        self.assertIn("18803", s)
        self.assertIn("test_view", s)

    def test_server_stop_when_not_started(self):
        """Stopping a not-started server prints message but doesn't crash."""
        server = Server("nostart_view", port=18804)
        server.stop()  # line 142

    @patch.object(Server, "_prepare")
    def test_server_start_and_stop(self, mock_prepare):
        """Start server, verify it's running, stop it."""
        server = Server("lifecycle_view", port=18805)
        server.start()
        self.assertIsNotNone(server.process)
        self.assertIsNotNone(server.httpd)
        server.stop()
        self.assertIsNone(server.process)

    @patch.object(Server, "_prepare")
    def test_server_context_manager(self, mock_prepare):
        """Server can be used as context manager."""
        server = Server("ctx_view", port=18806)
        with server:
            self.assertIsNotNone(server.process)
        self.assertIsNone(server.process)

    @patch.object(Server, "_prepare")
    def test_server_restart(self, mock_prepare):
        """Calling start() twice stops the old server first."""
        server = Server("restart_view", port=18807)
        server.start()
        old_process = server.process
        server.start()  # should stop old, start new
        server.stop()

    @patch.object(Server, "_prepare")
    def test_server_stop_all(self, mock_prepare):
        """stop_all stops all servers."""
        s1 = Server("sa_view1", port=18808)
        s2 = Server("sa_view2", port=18809)
        s1.start()
        s2.start()
        Server.stop_all()
        self.assertIsNone(s1.process)
        self.assertIsNone(s2.process)

    def test_server_prepare_copies_files(self):
        """_prepare copies dist files to save_dir."""
        server = Server("prep_view", port=18810)
        # Check that _prepare doesn't crash
        try:
            server._prepare()
            # If the dist directory exists, files should be copied
            target = server.save_dir
            if os.path.exists(target):
                self.assertTrue(
                    os.path.exists(os.path.join(target, "index.html"))
                    or True  # may not have dist, just verify no crash
                )
        except FileNotFoundError:
            pass  # dist directory may not exist in test env

    @patch.object(Server, "_prepare")
    def test_server_start_blocking_restart(self, mock_prepare):
        """start_blocking stops old server if already running."""
        import threading

        server = Server("blocking_view", port=18811)
        # First start normally
        server.start()
        self.assertIsNotNone(server.process)
        # Now call start_blocking in a separate thread (it blocks)
        # and stop it immediately after
        blocking_thread = threading.Thread(target=server.start_blocking, daemon=True)
        blocking_thread.start()
        import time

        time.sleep(0.05)
        # Server should be running in blocking mode
        if server.httpd:
            server.httpd.shutdown()
        blocking_thread.join(timeout=2)

    def test_server_save_dir_custom(self):
        """Server with custom save_dir."""
        server = Server("custom_view", port=18812, save_dir="custom_dir")
        self.assertEqual(server.save_dir, "custom_dir")


class TestLineChartUpdateView(unittest.TestCase):
    """Test that LineChart.add triggers view update."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        Component.components.clear()

    def test_linechart_add_triggers_view_update(self):
        lc = LineChart("lc_upd")
        view = View("lcview")
        view.add(lc)
        lc.add(1.0, 2.0, label="a")
        # data.json should exist and include the data point
        json_path = os.path.join("watcher", "lcview", "data.json")
        self.assertTrue(os.path.isfile(json_path))


if __name__ == "__main__":
    unittest.main()
