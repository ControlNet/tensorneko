import builtins
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tensorneko_util.backend.visual_lib import VisualLib, _VisualLibAvailability


class TestVisualLib(unittest.TestCase):
    _AVAIL_ATTRS = (
        "is_opencv_available",
        "is_torchvision_available",
        "is_matplotlib_available",
        "is_pil_available",
        "is_ffmpeg_available",
        "is_skimage_available",
    )

    def setUp(self):
        self._original_flags = {
            attr: getattr(_VisualLibAvailability, attr) for attr in self._AVAIL_ATTRS
        }
        for attr in self._AVAIL_ATTRS:
            setattr(_VisualLibAvailability, attr, None)

    def tearDown(self):
        for attr, value in self._original_flags.items():
            setattr(_VisualLibAvailability, attr, value)

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
        fake_torchvision = SimpleNamespace(__name__="torchvision")
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(
                replacements={"torchvision": fake_torchvision}
            ),
        ):
            self.assertTrue(VisualLib.pytorch_available())

        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(failing={"torchvision"}),
        ):
            self.assertTrue(VisualLib.pytorch_available())

    def test_pytorch_available_false_when_import_error(self):
        with patch(
            "builtins.__import__",
            side_effect=self._import_hook(failing={"torchvision"}),
        ):
            self.assertFalse(VisualLib.pytorch_available())

    def test_matplotlib_available_false_when_import_error(self):
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"matplotlib"})
        ):
            self.assertFalse(VisualLib.matplotlib_available())

    def test_pil_available_false_when_import_error(self):
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"PIL"})
        ):
            self.assertFalse(VisualLib.pil_available())

    def test_opencv_available_false_when_import_error(self):
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"cv2"})
        ):
            self.assertFalse(VisualLib.opencv_available())

    def test_ffmpeg_available_false_when_cli_missing(self):
        result = SimpleNamespace(returncode=1)
        with patch(
            "tensorneko_util.backend.visual_lib.subprocess.run", return_value=result
        ):
            self.assertFalse(VisualLib.ffmpeg_available())

    def test_ffmpeg_available_false_when_python_module_missing(self):
        result = SimpleNamespace(returncode=0)
        with patch(
            "tensorneko_util.backend.visual_lib.subprocess.run", return_value=result
        ):
            with patch(
                "builtins.__import__", side_effect=self._import_hook(failing={"ffmpeg"})
            ):
                self.assertFalse(VisualLib.ffmpeg_available())

    def test_ffmpeg_available_true_when_cli_and_python_module_exist(self):
        result = SimpleNamespace(returncode=0)
        fake_ffmpeg = SimpleNamespace(__name__="ffmpeg")
        with patch(
            "tensorneko_util.backend.visual_lib.subprocess.run", return_value=result
        ):
            with patch(
                "builtins.__import__",
                side_effect=self._import_hook(replacements={"ffmpeg": fake_ffmpeg}),
            ):
                self.assertTrue(VisualLib.ffmpeg_available())

    def test_skimage_available_false_when_import_error(self):
        with patch(
            "builtins.__import__", side_effect=self._import_hook(failing={"skimage"})
        ):
            self.assertFalse(VisualLib.skimage_available())


if __name__ == "__main__":
    unittest.main()
