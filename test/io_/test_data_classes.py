"""Tests for AudioData and VideoInfo __iter__ methods."""

import unittest

import numpy as np

from tensorneko_util.io.audio.audio_data import AudioData
from tensorneko_util.io.video.video_data import VideoInfo


class TestAudioDataIter(unittest.TestCase):
    def test_iter_unpacking(self):
        audio = np.zeros((1, 16000))
        ad = AudioData(audio=audio, sample_rate=16000)
        a, sr = ad
        np.testing.assert_array_equal(a, audio)
        self.assertEqual(sr, 16000)


class TestVideoInfoIter(unittest.TestCase):
    def test_iter_unpacking(self):
        vi = VideoInfo(video_fps=30.0, audio_fps=44100)
        vfps, afps = vi
        self.assertEqual(vfps, 30.0)
        self.assertEqual(afps, 44100)


if __name__ == "__main__":
    unittest.main()
