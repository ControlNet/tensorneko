import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torchaudio

from tensorneko_util.backend.audio_lib import AudioLib
from tensorneko_util.io.audio.audio_reader import AudioReader
from test.io_.audio._torchaudio_stub import load_wav_stub, save_wav_stub


class TestAudioReader(unittest.TestCase):
    def setUp(self):
        save_patch = patch("torchaudio.save", side_effect=save_wav_stub)
        load_patch = patch("torchaudio.load", side_effect=load_wav_stub)
        save_patch.start()
        load_patch.start()
        self.addCleanup(save_patch.stop)
        self.addCleanup(load_patch.stop)

    def _create_wav(
        self,
        directory: str,
        sample_rate: int = 16000,
        channels: int = 1,
        duration: int = 16000,
    ) -> tuple[str, torch.Tensor]:
        waveform = torch.randn(channels, duration)
        path = os.path.join(directory, "sample.wav")
        torchaudio.save(path, waveform, sample_rate)
        return path, waveform

    def test_read_wav_channel_first_shape_and_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path, waveform = self._create_wav(
                directory, sample_rate=16000, channels=2, duration=8000
            )
            audio = AudioReader.of(path, channel_first=True, backend=AudioLib.PYTORCH)

            self.assertEqual(tuple(audio.audio.shape), tuple(waveform.shape))
            self.assertEqual(audio.sample_rate, 16000)

    def test_read_wav_channel_last_shape_and_sample_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            _, waveform = self._create_wav(
                directory, sample_rate=22050, channels=2, duration=4000
            )
            path = os.path.join(directory, "sample.wav")
            audio = AudioReader.of(path, channel_first=False, backend=AudioLib.PYTORCH)

            expected_shape = (waveform.shape[1], waveform.shape[0])
            self.assertEqual(tuple(audio.audio.shape), expected_shape)
            self.assertEqual(audio.sample_rate, 22050)

    def test_read_wav_accepts_pathlib_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path, waveform = self._create_wav(
                directory, sample_rate=16000, channels=1, duration=1600
            )
            audio = AudioReader.of(
                Path(path), channel_first=True, backend=AudioLib.PYTORCH
            )

            self.assertEqual(tuple(audio.audio.shape), tuple(waveform.shape))
            self.assertEqual(audio.sample_rate, 16000)

    def test_read_nonexistent_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = os.path.join(directory, "missing.wav")
            with self.assertRaises(Exception):
                AudioReader.of(missing_path, backend=AudioLib.PYTORCH)

    def test_read_unknown_backend_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._create_wav(directory)
            with self.assertRaises(ValueError):
                AudioReader.of(path, backend="unknown_backend")

    def test_new_alias_matches_of(self):
        with tempfile.TemporaryDirectory() as directory:
            path, waveform = self._create_wav(
                directory, sample_rate=12000, channels=1, duration=1024
            )
            audio = AudioReader(path, backend=AudioLib.PYTORCH)

            self.assertEqual(tuple(audio.audio.shape), tuple(waveform.shape))
            self.assertEqual(audio.sample_rate, 12000)
