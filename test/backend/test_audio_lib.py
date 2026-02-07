import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tensorneko_util.backend.audio_lib import AudioLib
from tensorneko_util.backend.visual_lib import VisualLib


class TestAudioLib(unittest.TestCase):
    def setUp(self):
        class DummyAudioLib:
            _is_torchaudio_available = None

        self.dummy_cls = DummyAudioLib

    def _import_hook(self, failing=(), replacements=None):
        real_import = builtins.__import__
        failing = set(failing)
        replacements = replacements or {}

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            root_name = name.split(".")[0]
            if name in replacements:
                return replacements[name]
            if root_name in replacements:
                return replacements[root_name]
            if name in failing or root_name in failing:
                raise ImportError(f"mock missing module: {name}")
            return real_import(name, globals, locals, fromlist, level)

        return fake_import

    def test_pytorch_available_true_and_cached(self):
        fake_torchaudio = SimpleNamespace(__name__="torchaudio")
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(replacements={"torchaudio": fake_torchaudio}),
        ):
            self.assertTrue(AudioLib.pytorch_available.__func__(self.dummy_cls))

        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"torchaudio"})
        ):
            self.assertTrue(AudioLib.pytorch_available.__func__(self.dummy_cls))

    def test_pytorch_available_false_when_import_error(self):
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"torchaudio"})
        ):
            self.assertFalse(AudioLib.pytorch_available.__func__(self.dummy_cls))

    def test_ffmpeg_available_delegates_to_visual_lib_true(self):
        with patch.object(
            VisualLib, "ffmpeg_available", return_value=True
        ) as mock_ffmpeg:
            self.assertTrue(AudioLib.ffmpeg_available.__func__(self.dummy_cls))
            mock_ffmpeg.assert_called_once_with()

    def test_ffmpeg_available_delegates_to_visual_lib_false(self):
        with patch.object(
            VisualLib, "ffmpeg_available", return_value=False
        ) as mock_ffmpeg:
            self.assertFalse(AudioLib.ffmpeg_available.__func__(self.dummy_cls))
            mock_ffmpeg.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
