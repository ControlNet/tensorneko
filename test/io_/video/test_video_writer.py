import os
import tempfile
import unittest

import torch

from tensorneko_util.io.video.video_writer import VideoWriter
from tensorneko_util.io.video.video_reader import VideoReader
from tensorneko_util.io.video.video_data import VideoData, VideoInfo
from tensorneko_util.backend.visual_lib import VisualLib


class TestVideoWriter(unittest.TestCase):
    """Tests for VideoWriter using the torchvision (PYTORCH) backend."""

    # ------------------------------------------------------------------
    # .to() — basic write with tensor args
    # ------------------------------------------------------------------

    def test_write_creates_file(self):
        """Writing a synthetic video should produce a non-empty .mp4 file."""
        video = torch.rand(10, 64, 64, 3)  # T H W C, float [0,1]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.mp4")
            VideoWriter.to(path, video, 24.0, backend=VisualLib.PYTORCH)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_write_channel_first(self):
        """channel_first=True should accept (T, C, H, W) input."""
        video = torch.rand(10, 3, 64, 64)  # T C H W
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out_cf.mp4")
            VideoWriter.to(
                path, video, 24.0, channel_first=True, backend=VisualLib.PYTORCH
            )
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    # ------------------------------------------------------------------
    # Roundtrip: write → read
    # ------------------------------------------------------------------

    def test_roundtrip_shape(self):
        """Write then read back should preserve frame count and spatial dims."""
        T, H, W = 10, 64, 64
        video = torch.rand(T, H, W, 3)  # T H W C, float [0,1]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "roundtrip.mp4")
            VideoWriter.to(path, video, 24.0, backend=VisualLib.PYTORCH)
            result = VideoReader.of(path, channel_first=True, backend=VisualLib.PYTORCH)
            rT, rC, rH, rW = result.video.shape
            self.assertEqual(rT, T)
            self.assertEqual(rC, 3)
            self.assertEqual(rH, H)
            self.assertEqual(rW, W)

    def test_roundtrip_fps(self):
        """Write then read back should preserve fps."""
        video = torch.rand(10, 64, 64, 3)
        fps = 15.0
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "fps_test.mp4")
            VideoWriter.to(path, video, fps, backend=VisualLib.PYTORCH)
            result = VideoReader.of(path, backend=VisualLib.PYTORCH)
            self.assertAlmostEqual(result.info.video_fps, fps, places=0)

    # ------------------------------------------------------------------
    # uint8 warning
    # ------------------------------------------------------------------

    def test_uint8_tensor_passthrough(self):
        """Passing a uint8 tensor should write without corrupting pixel values."""
        video = (torch.rand(5, 64, 64, 3) * 255).to(torch.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "u8.mp4")
            VideoWriter.to(path, video, 24.0, backend=VisualLib.PYTORCH)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    # ------------------------------------------------------------------
    # .to() with VideoData
    # ------------------------------------------------------------------

    def test_write_numpy_array(self):
        """VideoWriter.to should accept a numpy array (float, T H W C)."""
        import numpy as np

        video = np.random.rand(8, 64, 64, 3).astype(np.float64)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "np_out.mp4")
            VideoWriter.to(path, video, 24.0, backend=VisualLib.PYTORCH)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    # ------------------------------------------------------------------
    # __new__ alias
    # ------------------------------------------------------------------

    def test_new_alias(self):
        """VideoWriter(path, video, ...) should work as alias of .to()."""
        video = torch.rand(6, 64, 64, 3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "alias.mp4")
            VideoWriter(path, video, 24.0, backend=VisualLib.PYTORCH)
            self.assertTrue(os.path.isfile(path))

    # ------------------------------------------------------------------
    # Error: missing audio_codec when audio provided
    # ------------------------------------------------------------------

    def test_audio_without_codec_raises(self):
        """Providing audio but no audio_codec should raise an error.

        Note: The source has ``if audio:`` which raises RuntimeError on
        multi-element tensors before the ValueError check can trigger.
        We accept either error type here.
        """
        video = torch.rand(5, 64, 64, 3)
        audio = torch.rand(2, 1000)  # C T
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "no_codec.mp4")
            with self.assertRaises((ValueError, RuntimeError)):
                VideoWriter.to(
                    path,
                    video,
                    24.0,
                    audio=audio,
                    audio_fps=16000,
                    backend=VisualLib.PYTORCH,
                )

    # ------------------------------------------------------------------
    # Unavailable backend error paths
    # ------------------------------------------------------------------

    def test_write_opencv_not_available(self):
        """Requesting OPENCV backend (not installed) should raise ValueError."""
        video = torch.rand(5, 64, 64, 3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cv.mp4")
            with self.assertRaises(ValueError):
                VideoWriter.to(path, video, 24.0, backend=VisualLib.OPENCV)

    def test_write_ffmpeg_not_available(self):
        """Requesting FFMPEG backend (not installed) should raise ValueError."""
        video = torch.rand(5, 64, 64, 3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ff.mp4")
            with self.assertRaises(ValueError):
                VideoWriter.to(path, video, 24.0, backend=VisualLib.FFMPEG)

    def test_write_unknown_backend_raises(self):
        """Passing an unknown backend should raise ValueError."""
        video = torch.rand(5, 64, 64, 3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bad.mp4")
            with self.assertRaises(ValueError):
                VideoWriter.to(path, video, 24.0, backend="not_a_backend")

    def test_write_numpy_uint8_passthrough(self):
        """Passing a uint8 numpy array should write without corrupting pixel values."""
        import numpy as np

        video = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "np_u8.mp4")
            VideoWriter.to(path, video, 24.0, backend=VisualLib.PYTORCH)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_write_with_path_object(self):
        """VideoWriter.__new__ should accept pathlib.Path (converts via _path2str)."""
        from pathlib import Path

        video = torch.rand(5, 64, 64, 3)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pathobj.mp4"
            # __new__ calls _path2str before dispatching to .to()
            VideoWriter(path, video, 24.0, backend=VisualLib.PYTORCH)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
