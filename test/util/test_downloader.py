import os
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from unittest.mock import patch, MagicMock

from tensorneko_util.util.downloader import (
    download_file,
    download_file_thread,
    download_files_thread,
)


class TestDownloadFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("tensorneko_util.util.downloader.urlretrieve")
    @patch("tensorneko_util.util.downloader._is_progress_bar_available", True)
    @patch("tensorneko_util.util.downloader.DownloadProgressBar")
    def test_download_to_dir(self, mock_pb_cls, mock_urlretrieve):
        """Test download_file extracts filename from URL and saves to dir_path."""
        mock_urlretrieve.return_value = (None, None)
        mock_pb = MagicMock()
        mock_pb.__enter__ = MagicMock(return_value=mock_pb)
        mock_pb.__exit__ = MagicMock(return_value=False)
        mock_pb_cls.return_value = mock_pb

        url = "http://example.com/data/myfile.zip"
        result = download_file(url, dir_path=self.tmpdir, progress_bar=True)
        expected = os.path.join(self.tmpdir, "myfile.zip")
        self.assertEqual(result, expected)
        mock_urlretrieve.assert_called_once()
        call_args = mock_urlretrieve.call_args
        self.assertEqual(call_args.kwargs["filename"], Path(expected))

    @patch("tensorneko_util.util.downloader.urlretrieve")
    @patch("tensorneko_util.util.downloader._is_progress_bar_available", True)
    @patch("tensorneko_util.util.downloader.DownloadProgressBar")
    def test_download_to_file_path(self, mock_pb_cls, mock_urlretrieve):
        """Test download_file with explicit file_path overrides dir_path."""
        mock_urlretrieve.return_value = (None, None)
        mock_pb = MagicMock()
        mock_pb.__enter__ = MagicMock(return_value=mock_pb)
        mock_pb.__exit__ = MagicMock(return_value=False)
        mock_pb_cls.return_value = mock_pb

        url = "http://example.com/data/myfile.zip"
        file_path = os.path.join(self.tmpdir, "subdir", "custom_name.zip")
        result = download_file(url, file_path=file_path, progress_bar=True)
        self.assertEqual(result, file_path)
        # Parent directory should be created
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "subdir")))
        mock_urlretrieve.assert_called_once()
        call_args = mock_urlretrieve.call_args
        self.assertEqual(call_args.kwargs["filename"], Path(file_path))

    @patch("tensorneko_util.util.downloader.urlretrieve")
    @patch("tensorneko_util.util.downloader._is_progress_bar_available", True)
    @patch("tensorneko_util.util.downloader.DownloadProgressBar")
    def test_download_with_progress_bar(self, mock_pb_cls, mock_urlretrieve):
        """Test download_file with progress bar enabled (True)."""
        mock_urlretrieve.return_value = (None, None)
        mock_pb = MagicMock()
        mock_pb.__enter__ = MagicMock(return_value=mock_pb)
        mock_pb.__exit__ = MagicMock(return_value=False)
        mock_pb_cls.return_value = mock_pb

        url = "http://example.com/data/file.tar.gz"
        result = download_file(url, dir_path=self.tmpdir, progress_bar=True)
        expected = os.path.join(self.tmpdir, "file.tar.gz")
        self.assertEqual(result, expected)
        # urlretrieve should be called with reporthook
        mock_urlretrieve.assert_called_once()
        call_kwargs = mock_urlretrieve.call_args
        self.assertIn("reporthook", call_kwargs.kwargs)
        # position should be None for progress_bar=True
        pb_call_kwargs = mock_pb_cls.call_args.kwargs
        self.assertIsNone(pb_call_kwargs.get("position"))

    @patch("tensorneko_util.util.downloader.urlretrieve")
    @patch("tensorneko_util.util.downloader._is_progress_bar_available", True)
    @patch("tensorneko_util.util.downloader.DownloadProgressBar")
    def test_download_with_progress_bar_position(self, mock_pb_cls, mock_urlretrieve):
        """Test download_file with progress bar position (int)."""
        mock_urlretrieve.return_value = (None, None)
        mock_pb = MagicMock()
        mock_pb.__enter__ = MagicMock(return_value=mock_pb)
        mock_pb.__exit__ = MagicMock(return_value=False)
        mock_pb_cls.return_value = mock_pb

        url = "http://example.com/data/file.bin"
        download_file(url, dir_path=self.tmpdir, progress_bar=3)
        # DownloadProgressBar should be called with position=3
        mock_pb_cls.assert_called_once()
        call_kwargs = mock_pb_cls.call_args
        self.assertEqual(call_kwargs.kwargs.get("position"), 3)

    @patch("tensorneko_util.util.downloader.urlretrieve")
    @patch("tensorneko_util.util.downloader._is_progress_bar_available", True)
    @patch("tensorneko_util.util.downloader.DownloadProgressBar")
    def test_download_creates_parent_dirs(self, mock_pb_cls, mock_urlretrieve):
        """Test that download_file creates parent directories."""
        mock_urlretrieve.return_value = (None, None)
        mock_pb = MagicMock()
        mock_pb.__enter__ = MagicMock(return_value=mock_pb)
        mock_pb.__exit__ = MagicMock(return_value=False)
        mock_pb_cls.return_value = mock_pb

        url = "http://example.com/nested/file.txt"
        nested_path = os.path.join(self.tmpdir, "a", "b", "c", "file.txt")
        download_file(url, file_path=nested_path, progress_bar=True)
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "a", "b", "c")))

    @patch("tensorneko_util.util.downloader._is_progress_bar_available", True)
    @patch("tensorneko_util.util.downloader.DownloadProgressBar")
    @patch(
        "tensorneko_util.util.downloader.urlretrieve",
        side_effect=OSError("Network error"),
    )
    def test_download_error_propagates(self, mock_urlretrieve, mock_pb_cls):
        """Test that download errors propagate to caller."""
        mock_pb = MagicMock()
        mock_pb.__enter__ = MagicMock(return_value=mock_pb)
        mock_pb.__exit__ = MagicMock(return_value=False)
        mock_pb_cls.return_value = mock_pb

        url = "http://example.com/bad/file.zip"
        with self.assertRaises(OSError):
            download_file(url, dir_path=self.tmpdir, progress_bar=True)

    @patch("tensorneko_util.util.downloader._is_progress_bar_available", False)
    def test_download_raises_without_tqdm(self):
        """Test that ImportError is raised when tqdm unavailable."""
        url = "http://example.com/data/file.zip"
        with self.assertRaises(ImportError):
            download_file(url, dir_path=self.tmpdir, progress_bar=True)


class TestDownloadFileThread(unittest.TestCase):
    @patch("tensorneko_util.util.downloader.download_file")
    def test_download_file_thread_returns_thread(self, mock_download):
        """Test download_file_thread returns a Thread."""
        t = download_file_thread("http://example.com/file.zip", progress_bar=False)
        self.assertIsInstance(t, Thread)

    @patch("tensorneko_util.util.downloader.download_file")
    def test_download_file_thread_calls_download(self, mock_download):
        """Test download_file_thread starts and invokes download_file."""
        t = download_file_thread(
            "http://example.com/file.zip", dir_path="/tmp", progress_bar=False
        )
        t.start()
        t.join(timeout=2)
        mock_download.assert_called_once_with(
            "http://example.com/file.zip", "/tmp", None, False
        )


class TestDownloadFilesThread(unittest.TestCase):
    @patch("tensorneko_util.util.downloader.download_file")
    def test_download_files_thread_returns_list(self, mock_download):
        """Test download_files_thread returns a list of Threads."""
        urls = ["http://example.com/a.zip", "http://example.com/b.zip"]
        threads = download_files_thread(urls, progress_bar=False)
        self.assertEqual(len(threads), 2)
        for t in threads:
            self.assertIsInstance(t, Thread)

    @patch("tensorneko_util.util.downloader.download_file")
    def test_download_files_thread_with_file_paths(self, mock_download):
        """Test download_files_thread with explicit file_paths."""
        urls = ["http://example.com/a.zip", "http://example.com/b.zip"]
        fps = ["/tmp/a.zip", "/tmp/b.zip"]
        threads = download_files_thread(urls, file_paths=fps, progress_bar=False)
        self.assertEqual(len(threads), 2)
        # Start and join all threads
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        self.assertEqual(mock_download.call_count, 2)

    @patch("tensorneko_util.util.downloader.download_file")
    def test_download_files_thread_with_progress_bars(self, mock_download):
        """Test download_files_thread assigns positional progress bars."""
        urls = [
            "http://example.com/a.zip",
            "http://example.com/b.zip",
            "http://example.com/c.zip",
        ]
        threads = download_files_thread(urls, progress_bar=True)
        self.assertEqual(len(threads), 3)
        # Start and join
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        # Verify progress bar positions: 0, 1, 2
        calls = mock_download.call_args_list
        self.assertEqual(calls[0].args[3], 0)
        self.assertEqual(calls[1].args[3], 1)
        self.assertEqual(calls[2].args[3], 2)


class TestDownloadNoProgressBar(unittest.TestCase):
    """Test download_file with progress_bar=False (previously broken path)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("tensorneko_util.util.downloader.urlretrieve")
    def test_download_no_progress_bar(self, mock_urlretrieve):
        """progress_bar=False should skip tqdm and call urlretrieve directly."""
        mock_urlretrieve.return_value = (None, None)
        url = "http://example.com/data/file.bin"
        result = download_file(url, dir_path=self.tmpdir, progress_bar=False)
        expected = os.path.join(self.tmpdir, "file.bin")
        self.assertEqual(result, expected)
        mock_urlretrieve.assert_called_once()
        # Should NOT have reporthook (no progress bar)
        call_kwargs = mock_urlretrieve.call_args
        self.assertNotIn("reporthook", call_kwargs.kwargs)


class TestDownloadProgressBarUpdateTo(unittest.TestCase):
    """Test DownloadProgressBar.update_to (lines 20-22)."""

    def test_update_to_with_tsize(self):
        """Lines 20-22: update_to sets total and updates."""
        from tensorneko_util.util.downloader import DownloadProgressBar

        pb = DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="test")
        pb.update_to(b=1, bsize=100, tsize=1000)
        self.assertEqual(pb.total, 1000)
        self.assertEqual(pb.n, 100)
        pb.close()

    def test_update_to_without_tsize(self):
        """Line 20: update_to without tsize doesn't set total."""
        from tensorneko_util.util.downloader import DownloadProgressBar

        pb = DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="test")
        pb.total = 500
        pb.update_to(b=2, bsize=50)
        self.assertEqual(pb.total, 500)  # unchanged
        self.assertEqual(pb.n, 100)
        pb.close()


if __name__ == "__main__":
    unittest.main()
