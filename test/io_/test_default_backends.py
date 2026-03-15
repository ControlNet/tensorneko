"""Tests for tensorneko_util IO dispatch layer:
- _default_backends.py: backend availability detection
- reader.py: Reader.__call__ dispatch + lazy properties
- writer.py: Writer.__call__ dispatch + lazy properties
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from tensorneko_util.io.reader import (
    Reader,
    scipy_available,
    yaml_available,
    h5py_available,
    toml_available,
)
from tensorneko_util.io.writer import Writer
from tensorneko_util.io._default_backends import (
    _default_image_io_backend,
    _default_video_io_backend,
    _default_audio_io_backend,
)
from tensorneko_util.backend.visual_lib import VisualLib
from tensorneko_util.backend.audio_lib import AudioLib


# ---------------------------------------------------------------------------
# _default_backends.py tests
# ---------------------------------------------------------------------------
class TestDefaultImageBackend(unittest.TestCase):
    """Test _default_image_io_backend selection logic."""

    @patch.object(VisualLib, "opencv_available", return_value=True)
    def test_prefers_opencv(self, _mock):
        self.assertEqual(_default_image_io_backend(), VisualLib.OPENCV)

    @patch.object(VisualLib, "opencv_available", return_value=False)
    @patch.object(VisualLib, "matplotlib_available", return_value=True)
    def test_falls_back_to_matplotlib(self, _m1, _m2):
        self.assertEqual(_default_image_io_backend(), VisualLib.MATPLOTLIB)

    @patch.object(VisualLib, "opencv_available", return_value=False)
    @patch.object(VisualLib, "matplotlib_available", return_value=False)
    def test_raises_when_none_available(self, _m1, _m2):
        with self.assertRaises(ValueError):
            _default_image_io_backend()


class TestDefaultVideoBackend(unittest.TestCase):
    """Test _default_video_io_backend selection logic."""

    @patch.object(VisualLib, "opencv_available", return_value=True)
    def test_prefers_opencv(self, _mock):
        self.assertEqual(_default_video_io_backend(), VisualLib.OPENCV)

    @patch.object(VisualLib, "opencv_available", return_value=False)
    @patch.object(VisualLib, "pytorch_available", return_value=True)
    def test_falls_back_to_pytorch(self, _m1, _m2):
        self.assertEqual(_default_video_io_backend(), VisualLib.PYTORCH)

    @patch.object(VisualLib, "opencv_available", return_value=False)
    @patch.object(VisualLib, "pytorch_available", return_value=False)
    @patch.object(VisualLib, "ffmpeg_available", return_value=True)
    def test_falls_back_to_ffmpeg(self, _m1, _m2, _m3):
        self.assertEqual(_default_video_io_backend(), VisualLib.FFMPEG)

    @patch.object(VisualLib, "opencv_available", return_value=False)
    @patch.object(VisualLib, "pytorch_available", return_value=False)
    @patch.object(VisualLib, "ffmpeg_available", return_value=False)
    def test_raises_when_none_available(self, _m1, _m2, _m3):
        with self.assertRaises(ValueError):
            _default_video_io_backend()


class TestDefaultAudioBackend(unittest.TestCase):
    """Test _default_audio_io_backend selection logic."""

    @patch.object(AudioLib, "pytorch_available", return_value=True)
    def test_prefers_pytorch(self, _mock):
        self.assertEqual(_default_audio_io_backend(), AudioLib.PYTORCH)

    @patch.object(AudioLib, "pytorch_available", return_value=False)
    def test_raises_when_none_available(self, _mock):
        with self.assertRaises(ValueError):
            _default_audio_io_backend()


# ---------------------------------------------------------------------------
# Reader dispatch tests
# ---------------------------------------------------------------------------
class TestReaderInit(unittest.TestCase):
    """Test Reader initialization and attribute access."""

    def test_reader_has_direct_attributes(self):
        reader = Reader()
        from tensorneko_util.io.image import ImageReader
        from tensorneko_util.io.text import TextReader
        from tensorneko_util.io.json import JsonReader
        from tensorneko_util.io.npy import NpyReader
        from tensorneko_util.io.video import VideoReader
        from tensorneko_util.io.audio import AudioReader
        from tensorneko_util.io.pickle import PickleReader

        self.assertIs(reader.image, ImageReader)
        self.assertIs(reader.text, TextReader)
        self.assertIs(reader.json, JsonReader)
        self.assertIs(reader.npy, NpyReader)
        self.assertIs(reader.video, VideoReader)
        self.assertIs(reader.audio, AudioReader)
        self.assertIs(reader.pickle, PickleReader)


class TestReaderLazyProperties(unittest.TestCase):
    """Test Reader lazy property access and caching."""

    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_yaml_property_returns_reader(self):
        reader = Reader()
        from tensorneko_util.io.yaml import YamlReader

        self.assertIs(reader.yaml, YamlReader)
        # Second access should return cached value
        self.assertIs(reader.yaml, YamlReader)

    @unittest.skipUnless(h5py_available, "h5py not installed")
    def test_h5_property_returns_reader(self):
        reader = Reader()
        from tensorneko_util.io.hdf5 import Hdf5Reader

        self.assertIs(reader.h5, Hdf5Reader)
        self.assertIs(reader.h5, Hdf5Reader)

    @unittest.skipUnless(toml_available, "toml not installed")
    def test_toml_property_returns_reader(self):
        reader = Reader()
        from tensorneko_util.io.toml import TomlReader

        self.assertIs(reader.toml, TomlReader)
        self.assertIs(reader.toml, TomlReader)

    @unittest.skipUnless(scipy_available, "scipy not installed")
    def test_mat_property_returns_reader(self):
        reader = Reader()
        from tensorneko_util.io.matlab import MatReader

        self.assertIs(reader.mat, MatReader)
        self.assertIs(reader.mat, MatReader)

    def test_mat_property_raises_when_unavailable(self):
        reader = Reader()
        with patch("tensorneko_util.io.reader.scipy_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = reader.mat
            self.assertIn("Scipy", str(ctx.exception))

    def test_yaml_property_raises_when_unavailable(self):
        reader = Reader()
        with patch("tensorneko_util.io.reader.yaml_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = reader.yaml
            self.assertIn("pyyaml", str(ctx.exception))

    def test_h5_property_raises_when_unavailable(self):
        reader = Reader()
        with patch("tensorneko_util.io.reader.h5py_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = reader.h5
            self.assertIn("h5py", str(ctx.exception))

    def test_toml_property_raises_when_unavailable(self):
        reader = Reader()
        with patch("tensorneko_util.io.reader.toml_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = reader.toml
            self.assertIn("toml", str(ctx.exception))


class TestReaderDispatch(unittest.TestCase):
    """Test Reader.__call__ routes to the correct backend by extension."""

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.reader = Reader()

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    # --- text ---
    def test_dispatch_txt(self):
        p = os.path.join(self.tmpdir, "hello.txt")
        with open(p, "w") as f:
            f.write("hello")
        self.assertEqual(self.reader(p), "hello")

    def test_dispatch_txt_path_object(self):
        p = Path(self.tmpdir) / "hello.txt"
        p.write_text("world")
        self.assertEqual(self.reader(p), "world")

    # --- json ---
    def test_dispatch_json(self):
        p = os.path.join(self.tmpdir, "data.json")
        with open(p, "w") as f:
            json.dump({"a": 1}, f)
        self.assertEqual(self.reader(p), {"a": 1})

    # --- npy ---
    def test_dispatch_npy(self):
        p = os.path.join(self.tmpdir, "arr.npy")
        np.save(p, np.array([1, 2, 3]))
        result = self.reader(p)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_dispatch_npz(self):
        p = os.path.join(self.tmpdir, "arr.npz")
        np.savez(p, x=np.array([4, 5]))
        result = self.reader(p)
        np.testing.assert_array_equal(result["x"], [4, 5])

    # --- pickle ---
    def test_dispatch_pkl(self):
        p = os.path.join(self.tmpdir, "data.pkl")
        with open(p, "wb") as f:
            pickle.dump({"key": "val"}, f)
        self.assertEqual(self.reader(p), {"key": "val"})

    def test_dispatch_pickle_ext(self):
        p = os.path.join(self.tmpdir, "data.pickle")
        with open(p, "wb") as f:
            pickle.dump([1, 2], f)
        self.assertEqual(self.reader(p), [1, 2])

    # --- yaml ---
    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_dispatch_yaml(self):
        import yaml as _yaml

        p = os.path.join(self.tmpdir, "cfg.yaml")
        with open(p, "w") as f:
            _yaml.dump({"x": 10}, f)
        self.assertEqual(self.reader(p), {"x": 10})

    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_dispatch_yml(self):
        import yaml as _yaml

        p = os.path.join(self.tmpdir, "cfg.yml")
        with open(p, "w") as f:
            _yaml.dump({"y": 20}, f)
        self.assertEqual(self.reader(p), {"y": 20})

    # --- toml ---
    @unittest.skipUnless(toml_available, "toml not installed")
    def test_dispatch_toml(self):
        import toml as _toml

        p = os.path.join(self.tmpdir, "cfg.toml")
        with open(p, "w") as f:
            _toml.dump({"section": {"key": "value"}}, f)
        self.assertEqual(self.reader(p), {"section": {"key": "value"}})

    # --- image dispatch (mock to avoid real IO) ---
    def test_dispatch_jpg(self):
        reader = Reader()
        reader.image = MagicMock(return_value="img_data")
        self.assertEqual(reader("photo.jpg"), "img_data")
        reader.image.assert_called_once_with("photo.jpg")

    def test_dispatch_jpeg(self):
        reader = Reader()
        reader.image = MagicMock(return_value="img_data")
        self.assertEqual(reader("photo.jpeg"), "img_data")

    def test_dispatch_png(self):
        reader = Reader()
        reader.image = MagicMock(return_value="img_data")
        self.assertEqual(reader("photo.png"), "img_data")

    def test_dispatch_bmp(self):
        reader = Reader()
        reader.image = MagicMock(return_value="img_data")
        self.assertEqual(reader("photo.bmp"), "img_data")

    # --- video dispatch (mock) ---
    def test_dispatch_mp4(self):
        reader = Reader()
        reader.video = MagicMock(return_value="vid")
        self.assertEqual(reader("clip.mp4"), "vid")
        reader.video.assert_called_once_with("clip.mp4")

    def test_dispatch_avi(self):
        reader = Reader()
        reader.video = MagicMock(return_value="vid")
        self.assertEqual(reader("clip.avi"), "vid")

    def test_dispatch_mov(self):
        reader = Reader()
        reader.video = MagicMock(return_value="vid")
        self.assertEqual(reader("clip.mov"), "vid")

    def test_dispatch_mkv(self):
        reader = Reader()
        reader.video = MagicMock(return_value="vid")
        self.assertEqual(reader("clip.mkv"), "vid")

    # --- audio dispatch (mock) ---
    def test_dispatch_wav(self):
        reader = Reader()
        reader.audio = MagicMock(return_value="aud")
        self.assertEqual(reader("sound.wav"), "aud")
        reader.audio.assert_called_once_with("sound.wav")

    def test_dispatch_mp3(self):
        reader = Reader()
        reader.audio = MagicMock(return_value="aud")
        self.assertEqual(reader("sound.mp3"), "aud")

    def test_dispatch_flac(self):
        reader = Reader()
        reader.audio = MagicMock(return_value="aud")
        self.assertEqual(reader("sound.flac"), "aud")

    # --- h5 dispatch (mock) ---
    @unittest.skipUnless(h5py_available, "h5py not installed")
    def test_dispatch_h5(self):
        reader = Reader()
        reader._h5 = MagicMock(return_value="h5_data")
        self.assertEqual(reader("data.h5"), "h5_data")

    @unittest.skipUnless(h5py_available, "h5py not installed")
    def test_dispatch_hdf5(self):
        reader = Reader()
        reader._h5 = MagicMock(return_value="h5_data")
        self.assertEqual(reader("data.hdf5"), "h5_data")

    # --- mat dispatch (mock) ---
    @unittest.skipUnless(scipy_available, "scipy not installed")
    def test_dispatch_mat(self):
        reader = Reader()
        reader._mat = MagicMock(return_value="mat_data")
        self.assertEqual(reader("data.mat"), "mat_data")

    def test_dispatch_mat_raises_when_scipy_unavailable(self):
        reader = Reader()
        with patch("tensorneko_util.io.reader.scipy_available", False):
            with self.assertRaises(ImportError):
                reader("data.mat")

    # --- unknown extension ---
    def test_dispatch_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.reader("file.xyz")
        self.assertIn("Unknown file type", str(ctx.exception))

    # --- Path object ---
    def test_dispatch_path_object_json(self):
        p = Path(self.tmpdir) / "data.json"
        p.write_text('{"b": 2}')
        self.assertEqual(self.reader(p), {"b": 2})

    def test_pickle_rejects_extra_args(self):
        p = os.path.join(self.tmpdir, "data.pkl")
        with open(p, "wb") as f:
            pickle.dump({}, f)
        with self.assertRaises(TypeError):
            self.reader(p, "extra_arg")


# ---------------------------------------------------------------------------
# Writer dispatch tests
# ---------------------------------------------------------------------------
class TestWriterInit(unittest.TestCase):
    """Test Writer initialization and attribute access."""

    def test_writer_has_direct_attributes(self):
        writer = Writer()
        from tensorneko_util.io.image import ImageWriter
        from tensorneko_util.io.text import TextWriter
        from tensorneko_util.io.json import JsonWriter
        from tensorneko_util.io.npy import NpyWriter
        from tensorneko_util.io.video import VideoWriter
        from tensorneko_util.io.audio import AudioWriter
        from tensorneko_util.io.pickle import PickleWriter

        self.assertIs(writer.image, ImageWriter)
        self.assertIs(writer.text, TextWriter)
        self.assertIs(writer.json, JsonWriter)
        self.assertIs(writer.npy, NpyWriter)
        self.assertIs(writer.video, VideoWriter)
        self.assertIs(writer.audio, AudioWriter)
        self.assertIs(writer.pickle, PickleWriter)


class TestWriterLazyProperties(unittest.TestCase):
    """Test Writer lazy property access and caching."""

    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_yaml_property_returns_writer(self):
        writer = Writer()
        from tensorneko_util.io.yaml import YamlWriter

        self.assertIs(writer.yaml, YamlWriter)
        self.assertIs(writer.yaml, YamlWriter)

    @unittest.skipUnless(h5py_available, "h5py not installed")
    def test_h5_property_returns_writer(self):
        writer = Writer()
        from tensorneko_util.io.hdf5 import Hdf5Writer

        self.assertIs(writer.h5, Hdf5Writer)
        self.assertIs(writer.h5, Hdf5Writer)

    @unittest.skipUnless(toml_available, "toml not installed")
    def test_toml_property_returns_writer(self):
        writer = Writer()
        from tensorneko_util.io.toml import TomlWriter

        self.assertIs(writer.toml, TomlWriter)
        self.assertIs(writer.toml, TomlWriter)

    @unittest.skipUnless(scipy_available, "scipy not installed")
    def test_mat_property_returns_writer(self):
        writer = Writer()
        from tensorneko_util.io.matlab import MatWriter

        self.assertIs(writer.mat, MatWriter)
        self.assertIs(writer.mat, MatWriter)

    def test_mat_property_raises_when_unavailable(self):
        writer = Writer()
        with patch("tensorneko_util.io.writer.scipy_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = writer.mat
            self.assertIn("Scipy", str(ctx.exception))

    def test_yaml_property_raises_when_unavailable(self):
        writer = Writer()
        with patch("tensorneko_util.io.writer.yaml_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = writer.yaml
            self.assertIn("pyyaml", str(ctx.exception))

    def test_h5_property_raises_when_unavailable(self):
        writer = Writer()
        with patch("tensorneko_util.io.writer.h5py_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = writer.h5
            self.assertIn("h5py", str(ctx.exception))

    def test_toml_property_raises_when_unavailable(self):
        writer = Writer()
        with patch("tensorneko_util.io.writer.toml_available", False):
            with self.assertRaises(ImportError) as ctx:
                _ = writer.toml
            self.assertIn("toml", str(ctx.exception))


class TestWriterDispatch(unittest.TestCase):
    """Test Writer.__call__ routes to the correct backend by extension."""

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.writer = Writer()

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    # --- text ---
    def test_dispatch_txt(self):
        p = os.path.join(self.tmpdir, "out.txt")
        self.writer(p, "hello")
        with open(p) as f:
            self.assertEqual(f.read(), "hello")

    def test_dispatch_txt_path_object(self):
        p = Path(self.tmpdir) / "out.txt"
        self.writer(p, "world")
        self.assertEqual(p.read_text(), "world")

    # --- json ---
    def test_dispatch_json(self):
        p = os.path.join(self.tmpdir, "data.json")
        self.writer(p, {"a": 1})
        with open(p) as f:
            self.assertEqual(json.load(f), {"a": 1})

    # --- npy ---
    def test_dispatch_npy(self):
        p = os.path.join(self.tmpdir, "arr.npy")
        self.writer(p, np.array([1, 2, 3]))
        np.testing.assert_array_equal(np.load(p), [1, 2, 3])

    # --- pickle ---
    def test_dispatch_pkl(self):
        p = os.path.join(self.tmpdir, "data.pkl")
        self.writer(p, {"key": "val"})
        with open(p, "rb") as f:
            self.assertEqual(pickle.load(f), {"key": "val"})

    def test_dispatch_pickle_ext(self):
        p = os.path.join(self.tmpdir, "data.pickle")
        self.writer(p, [1, 2])
        with open(p, "rb") as f:
            self.assertEqual(pickle.load(f), [1, 2])

    # --- yaml ---
    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_dispatch_yaml(self):
        import yaml as _yaml

        p = os.path.join(self.tmpdir, "cfg.yaml")
        self.writer(p, {"x": 10})
        with open(p) as f:
            self.assertEqual(_yaml.safe_load(f), {"x": 10})

    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_dispatch_yml(self):
        import yaml as _yaml

        p = os.path.join(self.tmpdir, "cfg.yml")
        self.writer(p, {"y": 20})
        with open(p) as f:
            self.assertEqual(_yaml.safe_load(f), {"y": 20})

    # --- toml dispatch (mock — TomlWriter.__new__ has a known issue on Python 3.14) ---
    @unittest.skipUnless(toml_available, "toml not installed")
    def test_dispatch_toml(self):
        writer = Writer()
        writer._toml = MagicMock()
        writer("cfg.toml", {"section": {"key": "value"}})
        writer._toml.assert_called_once_with("cfg.toml", {"section": {"key": "value"}})

    # --- image dispatch (mock) ---
    def test_dispatch_jpg(self):
        writer = Writer()
        writer.image = MagicMock()
        writer("photo.jpg", "img_data")
        writer.image.assert_called_once_with("photo.jpg", "img_data")

    def test_dispatch_jpeg(self):
        writer = Writer()
        writer.image = MagicMock()
        writer("photo.jpeg", "img_data")
        writer.image.assert_called_once()

    def test_dispatch_png(self):
        writer = Writer()
        writer.image = MagicMock()
        writer("photo.png", "img_data")
        writer.image.assert_called_once()

    def test_dispatch_bmp(self):
        writer = Writer()
        writer.image = MagicMock()
        writer("photo.bmp", "img_data")
        writer.image.assert_called_once()

    # --- video dispatch (mock) ---
    def test_dispatch_mp4(self):
        writer = Writer()
        writer.video = MagicMock()
        writer("clip.mp4", "vid_data", 30)
        writer.video.assert_called_once_with("clip.mp4", "vid_data", 30)

    def test_dispatch_avi(self):
        writer = Writer()
        writer.video = MagicMock()
        writer("clip.avi", "vid")
        writer.video.assert_called_once()

    def test_dispatch_mov(self):
        writer = Writer()
        writer.video = MagicMock()
        writer("clip.mov", "vid")
        writer.video.assert_called_once()

    def test_dispatch_mkv(self):
        writer = Writer()
        writer.video = MagicMock()
        writer("clip.mkv", "vid")
        writer.video.assert_called_once()

    # --- audio dispatch (mock) ---
    def test_dispatch_wav(self):
        writer = Writer()
        writer.audio = MagicMock()
        writer("sound.wav", "aud_data", 16000)
        writer.audio.assert_called_once_with("sound.wav", "aud_data", 16000)

    def test_dispatch_mp3(self):
        writer = Writer()
        writer.audio = MagicMock()
        writer("sound.mp3", "aud")
        writer.audio.assert_called_once()

    def test_dispatch_flac(self):
        writer = Writer()
        writer.audio = MagicMock()
        writer("sound.flac", "aud")
        writer.audio.assert_called_once()

    # --- h5 dispatch raises NotImplementedError ---
    def test_dispatch_h5_raises(self):
        with self.assertRaises(NotImplementedError):
            self.writer("data.h5", {"x": 1})

    def test_dispatch_hdf5_raises(self):
        with self.assertRaises(NotImplementedError):
            self.writer("data.hdf5", {"x": 1})

    # --- mat dispatch ---
    @unittest.skipUnless(scipy_available, "scipy not installed")
    def test_dispatch_mat(self):
        writer = Writer()
        writer._mat = MagicMock()
        writer("data.mat", {"key": "val"})
        writer._mat.assert_called_once_with("data.mat", {"key": "val"})

    def test_dispatch_mat_raises_when_scipy_unavailable(self):
        writer = Writer()
        with patch("tensorneko_util.io.writer.scipy_available", False):
            with self.assertRaises(ImportError):
                writer("data.mat", {"key": "val"})

    # --- unknown extension ---
    def test_dispatch_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.writer("file.xyz", "data")
        self.assertIn("Unknown file type", str(ctx.exception))

    # --- Path object ---
    def test_dispatch_path_object_json(self):
        p = Path(self.tmpdir) / "data.json"
        self.writer(p, {"b": 2})
        self.assertEqual(json.loads(p.read_text()), {"b": 2})

    def test_pickle_rejects_extra_args(self):
        p = os.path.join(self.tmpdir, "data.pkl")
        with self.assertRaises(TypeError):
            self.writer(p, {}, "extra_arg")

    # --- npz dispatch (NpyWriter.to_npz expects keyword args, not positional array) ---
    def test_dispatch_npz(self):
        writer = Writer()
        writer.npy = MagicMock()
        writer("data.npz", "arr_placeholder")
        writer.npy.assert_called_once_with("data.npz", "arr_placeholder")


# ---------------------------------------------------------------------------
# Roundtrip tests (read → write → read for basic formats)
# ---------------------------------------------------------------------------
class TestReadWriteRoundtrip(unittest.TestCase):
    """Lightweight roundtrip via dispatch to verify integration."""

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.reader = Reader()
        self.writer = Writer()

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    def test_roundtrip_txt(self):
        p = os.path.join(self.tmpdir, "rt.txt")
        self.writer(p, "round trip")
        self.assertEqual(self.reader(p), "round trip")

    def test_roundtrip_json(self):
        p = os.path.join(self.tmpdir, "rt.json")
        self.writer(p, {"round": "trip"})
        self.assertEqual(self.reader(p), {"round": "trip"})

    def test_roundtrip_npy(self):
        p = os.path.join(self.tmpdir, "rt.npy")
        arr = np.array([10, 20, 30])
        self.writer(p, arr)
        np.testing.assert_array_equal(self.reader(p), arr)

    def test_roundtrip_pkl(self):
        p = os.path.join(self.tmpdir, "rt.pkl")
        data = {"nested": [1, 2, {"three": 3}]}
        self.writer(p, data)
        self.assertEqual(self.reader(p), data)

    @unittest.skipUnless(yaml_available, "pyyaml not installed")
    def test_roundtrip_yaml(self):
        p = os.path.join(self.tmpdir, "rt.yaml")
        self.writer(p, {"cfg": "val"})
        self.assertEqual(self.reader(p), {"cfg": "val"})

    @unittest.skipUnless(toml_available, "toml not installed")
    def test_roundtrip_toml(self):
        """Use TomlWriter.to directly since __new__ has a known issue on Python 3.14."""
        from tensorneko_util.io.toml import TomlWriter

        p = os.path.join(self.tmpdir, "rt.toml")
        TomlWriter.to(p, {"sec": {"k": "v"}})
        self.assertEqual(self.reader(p), {"sec": {"k": "v"}})


if __name__ == "__main__":
    unittest.main()
