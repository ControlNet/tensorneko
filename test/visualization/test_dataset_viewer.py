"""Integration tests for DatasetVisualizer (FastAPI server)."""

import json
import time
import unittest
import urllib.request
import urllib.error

import torch
from torch.utils.data import TensorDataset, IterableDataset

try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401

    fastapi_available = True
except ImportError:
    fastapi_available = False

# Port base — each test class / test gets its own offset to avoid collisions.
_PORT_BASE = 18700


def _get(url: str, timeout: float = 5.0):
    """Perform a GET request and return (status_code, body_bytes)."""
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _get_json(url: str, timeout: float = 5.0):
    """GET and parse JSON body."""
    status, body = _get(url, timeout)
    return status, json.loads(body)


def _wait_for_server(
    port: int, host: str = "127.0.0.1", retries: int = 40, delay: float = 0.15
):
    """Block until the server responds on the given port."""
    url = f"http://{host}:{port}/api/info"
    for _ in range(retries):
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"Server on port {port} did not become ready")


# ---------------------------------------------------------------------------
# Synthetic datasets used by tests
# ---------------------------------------------------------------------------


def _make_image_label_dataset(n: int = 5):
    """TensorDataset of (3,32,32) images + scalar labels."""
    images = torch.rand(n, 3, 32, 32)
    labels = torch.arange(n)
    return TensorDataset(images, labels)


class _EmptyDataset(torch.utils.data.Dataset):
    """A dataset with zero length."""

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("empty")


class _BrokenDataset(torch.utils.data.Dataset):
    """Dataset whose __getitem__ always raises."""

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        raise RuntimeError("intentional error in __getitem__")


class _PartiallyBrokenDataset(torch.utils.data.Dataset):
    """Dataset where only index 0 works; others raise."""

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        if idx == 0:
            return {"image": torch.rand(3, 32, 32), "label": 0}
        raise RuntimeError(f"intentional error at index {idx}")


class _SimpleIterableDataset(IterableDataset):
    """An IterableDataset — has no __len__/__getitem__."""

    def __iter__(self):
        yield torch.tensor(1)


class _DictLabelDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._images = torch.rand(3, 3, 32, 32)
        self._labels = [0, 1, 0]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {"image": self._images[idx], "label": self._labels[idx]}


class _IntKeyDictLabelDataset(torch.utils.data.Dataset):
    def __init__(self):
        self._images = torch.rand(3, 3, 32, 32)

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {0: self._images[idx], 1: idx}


class _TensorLabelDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0:
            return {"logits": torch.tensor([0.1, 1.2, -0.3], dtype=torch.float32)}
        return {"logits": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)}


# ===========================================================================
# Test cases
# ===========================================================================


@unittest.skipUnless(fastapi_available, "fastapi/uvicorn not installed")
class TestServerLifecycle(unittest.TestCase):
    """1 & 2: start/stop and context manager."""

    # 1. Server starts and stops cleanly
    def test_start_stop(self):
        """Server starts and stops cleanly."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        ds = _make_image_label_dataset(3)
        port = _PORT_BASE + 0
        viz = DatasetVisualizer(ds, page_size=10)
        viz.start(port=port)
        try:
            _wait_for_server(port)
            status, _ = _get(f"http://127.0.0.1:{port}/api/info")
            self.assertEqual(status, 200)
        finally:
            viz.stop()

    # 2. Context manager works
    def test_context_manager(self):
        """Context manager starts and stops server."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        ds = _make_image_label_dataset(3)
        port = _PORT_BASE + 1
        viz = DatasetVisualizer(ds, page_size=10)
        # Manually start with explicit port, then use stop via __exit__
        viz.start(port=port)
        try:
            _wait_for_server(port)
            status, _ = _get(f"http://127.0.0.1:{port}/api/info")
            self.assertEqual(status, 200)
        finally:
            viz.__exit__(None, None, None)

        # After exit, server should be down (connection refused)
        time.sleep(0.3)
        with self.assertRaises(Exception):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/info", timeout=1)


@unittest.skipUnless(fastapi_available, "fastapi/uvicorn not installed")
class TestRoutes(unittest.TestCase):
    """Tests 3–10: all API routes on a 5-sample image+label dataset."""

    @classmethod
    def setUpClass(cls):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        cls.port = _PORT_BASE + 10
        cls.ds = _make_image_label_dataset(5)
        cls.viz = DatasetVisualizer(cls.ds, page_size=20)
        cls.viz.start(port=cls.port)
        _wait_for_server(cls.port)

    @classmethod
    def tearDownClass(cls):
        cls.viz.stop()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    # 3. GET /
    def test_get_root_html(self):
        """GET / → 200, HTML content."""
        status, body = _get(f"{self.base}/")
        self.assertEqual(status, 200)
        self.assertIn(b"Dataset Viewer", body)

    # 4. GET /api/info
    def test_api_info(self):
        """GET /api/info → JSON with length, schema, page_size."""
        status, data = _get_json(f"{self.base}/api/info")
        self.assertEqual(status, 200)
        self.assertEqual(data["length"], 5)
        self.assertIn("schema", data)
        self.assertEqual(data["page_size"], 20)
        # Schema should have two fields ("0" for image, "1" for label)
        schema = data["schema"]
        self.assertIn("0", schema)
        self.assertIn("1", schema)

    # 5. GET /api/samples?offset=0&limit=3
    def test_api_samples_limit(self):
        """GET /api/samples with limit → correct number of items."""
        status, data = _get_json(f"{self.base}/api/samples?offset=0&limit=3")
        self.assertEqual(status, 200)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 3)
        # Each item has idx and fields
        for item in data:
            self.assertIn("idx", item)
            self.assertIn("fields", item)

    # 6. GET /api/samples with pagination
    def test_api_samples_pagination(self):
        """Paginated samples return correct subset."""
        status, data = _get_json(f"{self.base}/api/samples?offset=3&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(len(data), 2)  # only indices 3,4 remain
        self.assertEqual(data[0]["idx"], 3)
        self.assertEqual(data[1]["idx"], 4)

    # 7. GET /api/sample/0
    def test_api_sample_single(self):
        """GET /api/sample/0 → 200, JSON with field metadata."""
        status, data = _get_json(f"{self.base}/api/sample/0")
        self.assertEqual(status, 200)
        self.assertIn("idx", data)
        self.assertEqual(data["idx"], 0)
        self.assertIn("fields", data)
        # Field "0" should be image type
        self.assertEqual(data["fields"]["0"]["type"], "image")

    # 8. GET /media/0/0 → PNG bytes
    def test_media_image(self):
        """GET /media/0/0 → 200, PNG image bytes."""
        status, body = _get(f"{self.base}/media/0/0")
        self.assertEqual(status, 200)
        # PNG magic bytes
        self.assertTrue(body[:4] == b"\x89PNG")

    # 9. GET /api/sample/999999 → 404
    def test_sample_out_of_range(self):
        """GET /api/sample/<out_of_range> → 404."""
        status, data = _get_json(f"{self.base}/api/sample/999999")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    # 10. GET /nonexistent → 404
    def test_nonexistent_path(self):
        """GET /nonexistent → 404."""
        status, _ = _get(f"{self.base}/nonexistent")
        self.assertEqual(status, 404)


@unittest.skipUnless(fastapi_available, "fastapi/uvicorn not installed")
class TestEdgeCases(unittest.TestCase):
    """Tests 11–16: edge cases."""

    # 11. IterableDataset → TypeError
    def test_iterable_dataset_rejected(self):
        """IterableDataset without __len__/__getitem__ → TypeError."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        with self.assertRaises(TypeError):
            DatasetVisualizer(_SimpleIterableDataset())

    # 12. Empty dataset → length=0
    def test_empty_dataset(self):
        """Empty dataset → /api/info shows length=0."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 20
        viz = DatasetVisualizer(_EmptyDataset(), page_size=10)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/info")
            self.assertEqual(status, 200)
            self.assertEqual(data["length"], 0)
            self.assertEqual(data["schema"], {})
        finally:
            viz.stop()

    # 13. __getitem__ exception → 500, server survives
    def test_getitem_exception_500(self):
        """Broken dataset → 500 JSON error, server keeps running."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 21
        viz = DatasetVisualizer(_BrokenDataset(), page_size=10)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)

            # Single sample → 500
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/0")
            self.assertEqual(status, 500)
            self.assertIn("error", data)

            # Server still alive
            status2, _ = _get_json(f"http://127.0.0.1:{port}/api/info")
            self.assertEqual(status2, 200)
        finally:
            viz.stop()

    # 14. User schema override → reflected in /api/info
    def test_user_schema_override(self):
        """User-provided schema override appears in /api/info."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 22
        ds = _make_image_label_dataset(3)
        custom_schema = {"0": "tensor", "1": "text"}
        viz = DatasetVisualizer(ds, schema=custom_schema, page_size=5)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)

            status, data = _get_json(f"http://127.0.0.1:{port}/api/info")
            self.assertEqual(status, 200)
            self.assertEqual(data["schema"]["0"], "tensor")
            self.assertEqual(data["schema"]["1"], "text")
            self.assertEqual(data["page_size"], 5)
        finally:
            viz.stop()

    # 15. Media for non-media field → 404
    def test_media_non_media_field_404(self):
        """GET /media/0/1 where field 1 is scalar (not image) → 404."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 23
        ds = _make_image_label_dataset(3)
        viz = DatasetVisualizer(ds, page_size=10)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, _ = _get(f"http://127.0.0.1:{port}/media/0/1")
            self.assertEqual(status, 404)
        finally:
            viz.stop()

    # 16. /api/samples with broken __getitem__ → skips errored, returns partial
    def test_api_samples_skip_errored(self):
        """Broken __getitem__ in samples endpoint → errors skipped silently."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 25
        viz = DatasetVisualizer(_PartiallyBrokenDataset(), page_size=20)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            # Fetch all 5 — only index 0 should succeed
            status, data = _get_json(
                f"http://127.0.0.1:{port}/api/samples?offset=0&limit=5"
            )
            self.assertEqual(status, 200)
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)  # only index 0 survives
            self.assertEqual(data[0]["idx"], 0)
        finally:
            viz.stop()

    # 17. /media with broken __getitem__ → 500
    def test_media_broken_getitem_500(self):
        """Broken __getitem__ for media endpoint → 500."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 26
        viz = DatasetVisualizer(_PartiallyBrokenDataset(), page_size=20)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            # Index 1 throws — media should return 500
            status, data = _get_json(f"http://127.0.0.1:{port}/media/1/image")
            self.assertEqual(status, 500)
            self.assertIn("error", data)
        finally:
            viz.stop()

    # 18. /media with field not in sample → 404
    def test_media_field_not_found_404(self):
        """GET /media/0/nonexistent_field → 404 (field not found)."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 27
        # Use _PartiallyBrokenDataset — index 0 works, schema has "image" as image type
        viz = DatasetVisualizer(
            _PartiallyBrokenDataset(),
            schema={"image": "image", "label": "scalar", "ghost": "image"},
            page_size=20,
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            # "ghost" is in schema as image but not in actual sample fields
            status, data = _get_json(f"http://127.0.0.1:{port}/media/0/ghost")
            self.assertEqual(status, 404)
            self.assertIn("error", data)
        finally:
            viz.stop()

    # 19. __enter__ returns self and starts server
    def test_enter_returns_self(self):
        """__enter__ calls start() and returns self."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 28
        ds = _make_image_label_dataset(3)
        viz = DatasetVisualizer(ds, page_size=10)
        # Call __enter__ which calls start() with default port
        returned = viz.__enter__()
        try:
            self.assertIs(returned, viz)
        finally:
            viz.__exit__(None, None, None)

    # 20. Media for out-of-range index → 404
    def test_media_out_of_range_404(self):
        """GET /media/999/0 → 404."""
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 24
        ds = _make_image_label_dataset(3)
        viz = DatasetVisualizer(ds, page_size=10)
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, _ = _get(f"http://127.0.0.1:{port}/media/999/0")
            self.assertEqual(status, 404)
        finally:
            viz.stop()


@unittest.skipUnless(fastapi_available, "fastapi/uvicorn not installed")
class TestLabelMappings(unittest.TestCase):
    def test_tuple_label_mapping_by_index_key(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 40
        ds = _make_image_label_dataset(3)
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={1: ["cat", "dog", "bird"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/2")
            self.assertEqual(status, 200)
            self.assertEqual(data["fields"]["1"]["type"], "scalar")
            self.assertEqual(data["fields"]["1"]["value"], 2)
            self.assertEqual(data["fields"]["1"]["label"], "bird")
            self.assertEqual(data["fields"]["1"]["label_index"], 2)
        finally:
            viz.stop()

    def test_dict_label_mapping_by_field_name(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 41
        ds = _DictLabelDataset()
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={"label": ["neg", "pos"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/1")
            self.assertEqual(status, 200)
            self.assertEqual(data["fields"]["label"]["type"], "scalar")
            self.assertEqual(data["fields"]["label"]["value"], 1)
            self.assertEqual(data["fields"]["label"]["label"], "pos")
            self.assertEqual(data["fields"]["label"]["label_index"], 1)
        finally:
            viz.stop()

    def test_dict_int_key_mapping_uses_raw_key(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 42
        ds = _IntKeyDictLabelDataset()
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={1: ["zero", "one", "two"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/2")
            self.assertEqual(status, 200)
            self.assertEqual(data["fields"]["1"]["type"], "scalar")
            self.assertEqual(data["fields"]["1"]["value"], 2)
            self.assertEqual(data["fields"]["1"]["label"], "two")
            self.assertEqual(data["fields"]["1"]["label_index"], 2)
        finally:
            viz.stop()

    def test_tuple_label_mapping_by_string_key(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 45
        ds = _make_image_label_dataset(3)
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={"1": ["cat", "dog", "bird"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/2")
            self.assertEqual(status, 200)
            self.assertEqual(data["fields"]["1"]["label"], "bird")
            self.assertEqual(data["fields"]["1"]["label_index"], 2)
        finally:
            viz.stop()

    def test_tensor_1d_label_mapping_uses_argmax(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 43
        ds = _TensorLabelDataset()
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={"logits": ["cat", "dog", "bird"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/0")
            self.assertEqual(status, 200)
            self.assertEqual(data["fields"]["logits"]["type"], "tensor")
            self.assertEqual(data["fields"]["logits"]["label"], "dog")
            self.assertEqual(data["fields"]["logits"]["label_index"], 1)
        finally:
            viz.stop()

    def test_missing_label_mapping_entry_keeps_original_metadata(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 44
        ds = _DictLabelDataset()
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={"label": ["neg"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(f"http://127.0.0.1:{port}/api/sample/1")
            self.assertEqual(status, 200)
            self.assertEqual(data["fields"]["label"]["type"], "scalar")
            self.assertEqual(data["fields"]["label"]["value"], 1)
            self.assertNotIn("label", data["fields"]["label"])
            self.assertNotIn("label_index", data["fields"]["label"])
        finally:
            viz.stop()

    def test_api_samples_contains_mapped_labels(self):
        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        port = _PORT_BASE + 46
        ds = _DictLabelDataset()
        viz = DatasetVisualizer(
            ds,
            page_size=10,
            label_mappings={"label": ["neg", "pos"]},
        )
        viz.start(port=port)
        try:
            _wait_for_server(port, retries=40)
            status, data = _get_json(
                f"http://127.0.0.1:{port}/api/samples?offset=0&limit=3"
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]["fields"]["label"]["label"], "neg")
            self.assertEqual(data[0]["fields"]["label"]["label_index"], 0)
            self.assertEqual(data[1]["fields"]["label"]["label"], "pos")
            self.assertEqual(data[1]["fields"]["label"]["label_index"], 1)
        finally:
            viz.stop()


if __name__ == "__main__":
    unittest.main()
