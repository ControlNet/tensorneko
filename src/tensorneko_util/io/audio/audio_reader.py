from typing import Optional, Union
from pathlib import Path

import numpy as np

from .audio_data import AudioData
from .._default_backends import _default_audio_io_backend
from .._path_conversion import _path2str
from ...backend.audio_lib import AudioLib


class AudioReader:
    """AudioReader for reading audio file"""

    @staticmethod
    def of(path: Union[str, Path], channel_first: bool = True, backend: Optional[AudioLib] = None) -> AudioData:
        """
        Read audio tensor from given file.

        Args:
            path (``str`` | ``pathlib.Path``): Path to the audio file.
            channel_first (``bool``, optional): Whether the audio is channel first. The output shape is (C, T) if true
                and (T, C) if false. Default: True.
            backend (:class:`~tensorneko.io.audio.audio_lib.AudioLib`, optional): The audio library to use.
                Default: pytorch.

        Returns:
            :class:`~.audio_data.AudioData`: The Audio data in the file.
        """
        backend = backend or _default_audio_io_backend()
        path = _path2str(path)

        if backend == AudioLib.PYTORCH:
            if not AudioLib.pytorch_available():
                raise ValueError("Torchaudio is not available.")
            import torchaudio
            return AudioData(*torchaudio.load(path, channels_first=channel_first))
        elif backend == AudioLib.FFMPEG:
            if not AudioLib.ffmpeg_available():
                raise ValueError("FFmpeg is not available.")
            import ffmpeg

            for stream in ffmpeg.probe(path)["streams"]:
                if stream["codec_type"] == "audio":
                    sample_rate = int(stream["sample_rate"])
                    channel = int(stream["channels"])
                    break
            else:
                raise RuntimeError("No audio stream found.")

            try:
                out, _ = (
                    ffmpeg.input(path, threads=0)
                    .output("-", format="s16le", acodec="pcm_s16le")
                    .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

            arr = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            arr = arr.reshape(-1, channel)
            arr = arr.T if channel_first else arr

            return AudioData(arr, sample_rate)
        else:
            raise ValueError("Unknown audio library: {}".format(backend))

    def __new__(cls, path: Union[str, Path], channel_first: bool = True, backend: Optional[AudioLib] = None) -> AudioData:
        """Alias of :meth:`~AudioReader.of`"""
        return cls.of(path, channel_first, backend)
