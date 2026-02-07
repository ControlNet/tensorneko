import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from tensorneko_util.backend.audio_lib import AudioLib
from tensorneko_util.io.audio.audio_data import AudioData
from tensorneko_util.io.audio.audio_reader import AudioReader
from tensorneko_util.io.audio.audio_writer import AudioWriter
from test.io_.audio._torchaudio_stub import load_wav_stub, save_wav_stub


class TestAudioWriter(unittest.TestCase):
    def setUp(self):
        save_patch = patch("torchaudio.save", side_effect=save_wav_stub)
        load_patch = patch("torchaudio.load", side_effect=load_wav_stub)
        save_patch.start()
        load_patch.start()
        self.addCleanup(save_patch.stop)
        self.addCleanup(load_patch.stop)

    def _random_audio(self, channels: int = 1, duration: int = 16000) -> torch.Tensor:
        return torch.randn(channels, duration)

    def test_write_and_readback_shape_matches(self):
        audio = self._random_audio(channels=2, duration=4000)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "roundtrip.wav")
            AudioWriter.to(
                path,
                audio,
                sample_rate=16000,
                channel_first=True,
                backend=AudioLib.PYTORCH,
            )

            loaded = AudioReader.of(path, channel_first=True, backend=AudioLib.PYTORCH)
            self.assertEqual(tuple(loaded.audio.shape), tuple(audio.shape))
            self.assertEqual(loaded.sample_rate, 16000)

    def test_write_audio_data_dispatch_roundtrip(self):
        audio_tensor = self._random_audio(channels=1, duration=2000)
        audio_data = AudioData(audio_tensor, 11025)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "audio_data.wav")
            AudioWriter.to(path, audio_data, backend=AudioLib.PYTORCH)

            loaded = AudioReader.of(path, backend=AudioLib.PYTORCH)
            self.assertEqual(tuple(loaded.audio.shape), tuple(audio_tensor.shape))
            self.assertEqual(loaded.sample_rate, 11025)

    def test_write_different_sample_rates(self):
        audio = self._random_audio(channels=1, duration=1600)
        with tempfile.TemporaryDirectory() as directory:
            for sample_rate in (8000, 16000, 22050):
                with self.subTest(sample_rate=sample_rate):
                    path = os.path.join(directory, f"sample_{sample_rate}.wav")
                    AudioWriter.to(
                        path, audio, sample_rate=sample_rate, backend=AudioLib.PYTORCH
                    )

                    loaded = AudioReader.of(path, backend=AudioLib.PYTORCH)
                    self.assertEqual(tuple(loaded.audio.shape), tuple(audio.shape))
                    self.assertEqual(loaded.sample_rate, sample_rate)

    def test_write_unknown_backend_raises_value_error(self):
        audio = self._random_audio()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unknown_backend.wav")
            with self.assertRaises(ValueError):
                AudioWriter.to(path, audio, sample_rate=16000, backend=AudioLib.FFMPEG)

    def test_pytorch_unavailable_raises(self):
        """Cover audio_writer.py line 54: pytorch_available() returns False."""
        audio = self._random_audio()
        with patch.object(AudioLib, "pytorch_available", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                AudioWriter.to(
                    "dummy.wav", audio, sample_rate=16000, backend=AudioLib.PYTORCH
                )
            self.assertIn("Torchaudio", str(ctx.exception))

    def test_new_alias_writes_file(self):
        audio = self._random_audio(channels=1, duration=1000)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "via_new.wav")
            AudioWriter(path, audio, sample_rate=16000, backend=AudioLib.PYTORCH)

            loaded = AudioReader.of(path, backend=AudioLib.PYTORCH)
            self.assertEqual(tuple(loaded.audio.shape), tuple(audio.shape))
            self.assertEqual(loaded.sample_rate, 16000)
