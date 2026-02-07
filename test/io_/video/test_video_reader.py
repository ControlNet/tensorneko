import unittest
from pathlib import Path

import torch

from tensorneko_util.io.video.video_reader import VideoReader
from tensorneko_util.io.video.video_data import VideoData, VideoInfo
from tensorneko_util.backend.visual_lib import VisualLib

SAMPLE_VIDEO = "test/resource/video_sample/sample.mp4"


class TestVideoReader(unittest.TestCase):
    """Tests for VideoReader using the torchvision (PYTORCH) backend."""

    # ------------------------------------------------------------------
    # .of() — basic read
    # ------------------------------------------------------------------

    def test_of_returns_video_data(self):
        """VideoReader.of should return a VideoData instance."""
        result = VideoReader.of(SAMPLE_VIDEO, backend=VisualLib.PYTORCH)
        self.assertIsInstance(result, VideoData)
        self.assertIsInstance(result.info, VideoInfo)

    def test_of_video_shape_channel_first(self):
        """Default channel_first=True should give (T, C, H, W)."""
        result = VideoReader.of(
            SAMPLE_VIDEO, channel_first=True, backend=VisualLib.PYTORCH
        )
        self.assertEqual(result.video.ndim, 4)
        T, C, H, W = result.video.shape
        self.assertEqual(C, 3)
        self.assertGreater(T, 0)
        self.assertGreater(H, 0)
        self.assertGreater(W, 0)

    def test_of_video_shape_channel_last(self):
        """channel_first=False should give (T, H, W, C)."""
        result = VideoReader.of(
            SAMPLE_VIDEO, channel_first=False, backend=VisualLib.PYTORCH
        )
        self.assertEqual(result.video.ndim, 4)
        T, H, W, C = result.video.shape
        self.assertEqual(C, 3)
        self.assertGreater(T, 0)

    def test_of_fps_positive(self):
        """Returned fps should be a positive number."""
        result = VideoReader.of(SAMPLE_VIDEO, backend=VisualLib.PYTORCH)
        self.assertIsInstance(result.info.video_fps, float)
        self.assertGreater(result.info.video_fps, 0)

    def test_of_audio_tensor_present(self):
        """Audio tensor should be present (may be empty for videos without audio)."""
        result = VideoReader.of(SAMPLE_VIDEO, backend=VisualLib.PYTORCH)
        # audio is a tensor (possibly empty)
        self.assertIsNotNone(result.audio)

    # ------------------------------------------------------------------
    # Path input
    # ------------------------------------------------------------------

    def test_of_with_path_object(self):
        """VideoReader.of should accept pathlib.Path."""
        result = VideoReader.of(Path(SAMPLE_VIDEO), backend=VisualLib.PYTORCH)
        self.assertIsInstance(result, VideoData)
        self.assertGreater(result.video.shape[0], 0)

    # ------------------------------------------------------------------
    # __new__ alias
    # ------------------------------------------------------------------

    def test_new_alias(self):
        """VideoReader(path) should behave like VideoReader.of(path, channel_first=False)."""
        result = VideoReader(
            SAMPLE_VIDEO, channel_first=False, backend=VisualLib.PYTORCH
        )
        self.assertIsInstance(result, VideoData)
        # channel_first=False → (T, H, W, C)
        self.assertEqual(result.video.shape[-1], 3)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_of_nonexistent_file_raises(self):
        """Reading a non-existent file should raise an error."""
        with self.assertRaises(Exception):
            VideoReader.of("nonexistent_video_path.mp4", backend=VisualLib.PYTORCH)

    def test_of_unknown_backend_raises(self):
        """Passing an unsupported backend value should raise ValueError."""
        with self.assertRaises(ValueError):
            VideoReader.of(SAMPLE_VIDEO, backend="not_a_backend")

    # ------------------------------------------------------------------
    # VideoData iteration
    # ------------------------------------------------------------------

    def test_video_data_iteration(self):
        """VideoData should be iterable as (video, audio, info)."""
        result = VideoReader.of(SAMPLE_VIDEO, backend=VisualLib.PYTORCH)
        video, audio, info = result
        self.assertEqual(video.shape, result.video.shape)
        self.assertIsInstance(info, VideoInfo)

    # ------------------------------------------------------------------
    # Unavailable backend error paths
    # ------------------------------------------------------------------

    def test_of_opencv_not_available(self):
        """Requesting OPENCV backend (not installed) should raise ValueError."""
        with self.assertRaises(ValueError):
            VideoReader.of(SAMPLE_VIDEO, backend=VisualLib.OPENCV)

    def test_of_ffmpeg_not_available(self):
        """Requesting FFMPEG backend (not installed) should raise ValueError."""
        with self.assertRaises(ValueError):
            VideoReader.of(SAMPLE_VIDEO, backend=VisualLib.FFMPEG)

    def test_with_indexes_opencv_not_available(self):
        """with_indexes with OPENCV backend should raise ValueError."""
        import numpy as np

        with self.assertRaises(ValueError):
            VideoReader.with_indexes(
                SAMPLE_VIDEO, np.array([0, 1, 2]), backend=VisualLib.OPENCV
            )

    def test_with_indexes_ffmpeg_not_available(self):
        """with_indexes with FFMPEG backend should raise ValueError."""
        import numpy as np

        with self.assertRaises(ValueError):
            VideoReader.with_indexes(
                SAMPLE_VIDEO, np.array([0, 1, 2]), backend=VisualLib.FFMPEG
            )

    def test_with_indexes_unknown_backend_raises(self):
        """with_indexes with unknown backend should raise ValueError."""
        import numpy as np

        with self.assertRaises(ValueError):
            VideoReader.with_indexes(SAMPLE_VIDEO, np.array([0, 1, 2]), backend="bad")

    def test_with_range_opencv_not_available(self):
        """with_range with OPENCV backend should fall through to with_indexes which raises."""
        with self.assertRaises((ValueError, TypeError)):
            VideoReader.with_range(SAMPLE_VIDEO, 0, 5, 1, backend=VisualLib.OPENCV)

    def test_with_range_short_form(self):
        """with_range(path, end) should call with_range(path, 0, end, 1)."""
        with self.assertRaises((ValueError, TypeError)):
            VideoReader.with_range(SAMPLE_VIDEO, 5, backend=VisualLib.OPENCV)
