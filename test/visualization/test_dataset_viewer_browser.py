"""Playwright browser tests for the Dataset Viewer frontend.

These tests require playwright to be installed::

    pip install playwright
    playwright install chromium

All tests are skipped gracefully if playwright is not available.
"""

import socket
import time
import unittest
import urllib.request

# ---------------------------------------------------------------------------
# Optional dependency checks (must not crash at import time)
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    import torch  # noqa: F401
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401

    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False

# Port base — offset from the API-test base to avoid collisions.
_PORT_BASE = 18800


def _find_free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port, host="127.0.0.1", retries=40, delay=0.15):
    """Block until the server responds on the given port."""
    url = f"http://{host}:{port}/api/info"
    for _ in range(retries):
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"Server on port {port} did not become ready")


# ===========================================================================
# Browser test case
# ===========================================================================


@unittest.skipUnless(PLAYWRIGHT_AVAILABLE, "playwright not installed")
@unittest.skipUnless(SERVER_AVAILABLE, "torch/fastapi/uvicorn not available")
class TestDatasetViewerBrowser(unittest.TestCase):
    """Browser-based integration tests for the Dataset Viewer frontend."""

    @classmethod
    def setUpClass(cls):
        """Start server with a synthetic 40-sample dataset (2 pages)."""
        import torch
        from torch.utils.data import TensorDataset

        from tensorneko.visualization.dataset_viewer.server import DatasetVisualizer

        images = torch.rand(40, 3, 32, 32)
        labels = torch.arange(40)
        cls.ds = TensorDataset(images, labels)

        cls.port = _find_free_port()
        cls.viz = DatasetVisualizer(cls.ds, page_size=20)
        cls.viz.start(port=cls.port)
        _wait_for_server(cls.port)
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        # Launch headless Chromium
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        """Shut down browser and server."""
        cls.browser.close()
        cls.pw.stop()
        cls.viz.stop()

    def setUp(self):
        """Create a fresh browser page for each test."""
        self.page = self.browser.new_page()

    def tearDown(self):
        """Close the page after each test."""
        self.page.close()

    # ------------------------------------------------------------------
    # 1. Page load
    # ------------------------------------------------------------------

    def test_page_loads_with_samples(self):
        """Page loads and renders the first page of 20 sample items."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        items = self.page.query_selector_all(".sample-item")
        self.assertEqual(len(items), 20)
        # Dataset info should mention 40 total samples
        info_text = self.page.text_content("#dataset-info")
        self.assertIn("40", info_text)

    # ------------------------------------------------------------------
    # 2. Sample selection
    # ------------------------------------------------------------------

    def test_sample_selection_shows_detail(self):
        """Clicking a sample shows the detail panel with field cards."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        self.page.click(".sample-item")
        self.page.wait_for_selector(".field-card", timeout=10000)
        cards = self.page.query_selector_all(".field-card")
        self.assertGreater(len(cards), 0)

    # ------------------------------------------------------------------
    # 3. Pagination
    # ------------------------------------------------------------------

    def test_pagination_next(self):
        """Clicking 'Next' advances to page 2."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        self.page.click("#btn-next")
        # Wait for page 2 text to appear
        self.page.wait_for_function(
            "document.getElementById('page-info').textContent.includes('2')",
            timeout=5000,
        )
        page_text = self.page.text_content("#page-info")
        self.assertIn("2", page_text)

    # ------------------------------------------------------------------
    # 4. Jump navigation
    # ------------------------------------------------------------------

    def test_index_jump_navigation(self):
        """Jump input navigates to the target sample across pages."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        self.page.fill("#jump-input", "25")
        self.page.click("#btn-jump")
        # Sample 25 is on page 2 — wait for page change and active item
        self.page.wait_for_selector('.sample-item.active[data-idx="25"]', timeout=10000)
        active = self.page.query_selector(".sample-item.active")
        self.assertIsNotNone(active)
        self.assertEqual(active.get_attribute("data-idx"), "25")

    # ------------------------------------------------------------------
    # 5. Dark mode (via CSS media emulation)
    # ------------------------------------------------------------------

    def test_dark_mode_via_emulation(self):
        """Dark mode activates when prefers-color-scheme is dark."""
        self.page.emulate_media(color_scheme="dark")
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        bg = self.page.evaluate("getComputedStyle(document.body).backgroundColor")
        # Dark bg is rgb(26, 26, 26) from --bg: #1a1a1a
        self.assertIn("26", bg)

    # ------------------------------------------------------------------
    # 6. Responsive layout
    # ------------------------------------------------------------------

    def test_responsive_layout_stacks_vertically(self):
        """At 375px width the .app container uses flex-direction: column."""
        self.page.set_viewport_size({"width": 375, "height": 667})
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        direction = self.page.evaluate(
            "getComputedStyle(document.querySelector('.app')).flexDirection"
        )
        self.assertEqual(direction, "column")

    # ------------------------------------------------------------------
    # 7. Image zoom modal
    # ------------------------------------------------------------------

    def test_image_zoom_modal(self):
        """Clicking an image opens a zoom overlay; Escape closes it."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        self.page.click(".sample-item")
        self.page.wait_for_selector(".field-image img", timeout=10000)
        self.page.click(".field-image img")
        self.page.wait_for_selector(".zoom-overlay", timeout=5000)
        overlay = self.page.query_selector(".zoom-overlay")
        self.assertIsNotNone(overlay)
        # Close with Escape
        self.page.keyboard.press("Escape")
        self.page.wait_for_selector(".zoom-overlay", state="detached", timeout=5000)
        overlay = self.page.query_selector(".zoom-overlay")
        self.assertIsNone(overlay)

    # ------------------------------------------------------------------
    # 8. Collapsible field cards
    # ------------------------------------------------------------------

    def test_collapsible_field_cards(self):
        """Clicking a field card header collapses its body."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        self.page.click(".sample-item")
        self.page.wait_for_selector(".field-card-header", timeout=10000)
        # Initially expanded
        header = self.page.query_selector(".field-card-header")
        self.assertEqual(header.get_attribute("aria-expanded"), "true")
        # Click to collapse
        self.page.click(".field-card-header")
        self.assertEqual(header.get_attribute("aria-expanded"), "false")
        body = self.page.query_selector(".field-card-body")
        body_class = body.get_attribute("class") or ""
        self.assertIn("collapsed", body_class)

    # ------------------------------------------------------------------
    # 9. Keyboard navigation — ArrowDown in sample list
    # ------------------------------------------------------------------

    def test_keyboard_navigation_arrows(self):
        """ArrowDown in sample list moves focus to the next item."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        # Focus the first item
        first_item = self.page.query_selector(".sample-item")
        first_item.focus()
        # Press ArrowDown
        self.page.keyboard.press("ArrowDown")
        self.page.wait_for_timeout(300)
        # The second item should now be focused
        items = self.page.query_selector_all(".sample-item")
        focused_idx = self.page.evaluate(
            "document.activeElement.getAttribute('data-idx')"
        )
        second_idx = items[1].get_attribute("data-idx")
        self.assertEqual(focused_idx, second_idx)

    # ------------------------------------------------------------------
    # 10. Keyboard navigation — j/k shortcuts
    # ------------------------------------------------------------------

    def test_keyboard_jk_navigation(self):
        """j key navigates to the next sample."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        # Select first sample
        self.page.click(".sample-item")
        self.page.wait_for_selector(".field-card", timeout=10000)
        # Focus the detail panel (not an input) so j/k work
        self.page.click(".panel-right")
        self.page.keyboard.press("j")
        # Wait for the next sample to become active
        self.page.wait_for_selector('.sample-item.active[data-idx="1"]', timeout=5000)
        active = self.page.query_selector(".sample-item.active")
        self.assertIsNotNone(active)
        self.assertEqual(active.get_attribute("data-idx"), "1")

    # ------------------------------------------------------------------
    # 11. Collapse-all / Expand-all toggle
    # ------------------------------------------------------------------

    def test_collapse_all_expand_all(self):
        """'Collapse All' button collapses every card, 'Expand All' restores."""
        self.page.goto(self.base_url)
        self.page.wait_for_selector(".sample-item", timeout=10000)
        self.page.click(".sample-item")
        self.page.wait_for_selector(".collapse-toggle-btn", timeout=10000)
        # Click Collapse All
        self.page.click(".collapse-toggle-btn")
        bodies = self.page.query_selector_all(".field-card-body")
        for body in bodies:
            self.assertIn("collapsed", body.get_attribute("class") or "")
        # Click Expand All
        self.page.click(".collapse-toggle-btn")
        for body in self.page.query_selector_all(".field-card-body"):
            self.assertNotIn("collapsed", body.get_attribute("class") or "")


if __name__ == "__main__":
    unittest.main()
