from typing import Optional

from .audio_data import AudioData
from .._default_backends import _default_audio_io_backend
from ...backend.audio_lib import AudioLib


class AudioReader:
    """AudioReader for reading audio file"""

    @staticmethod
    def of(path: str, channel_first: bool = True, backend: Optional[AudioLib] = None) -> AudioData:
        """
        Read audio tensor from given file.

        Args:
            path (``str``): Path to the audio file.
            channel_first (``bool``, optional): Whether the audio is channel first. The output shape is (C, T) if true
                and (T, C) if false. Default: True.
            backend (:class:`~tensorneko.io.audio.audio_lib.AudioLib`, optional): The audio library to use.
                Default: pytorch.

        Returns:
            :class:`~.audio_data.AudioData`: The Audio data in the file.
        """
        backend = backend or _default_audio_io_backend()

        if backend == AudioLib.PYTORCH:
            if not AudioLib.pytorch_available():
                raise ValueError("Torchaudio is not available.")
            import torchaudio
            return AudioData(*torchaudio.load(path, channels_first=channel_first))
        else:
            raise ValueError("Unknown audio library: {}".format(backend))

    def __new__(cls, path: str, channel_first: bool = True, backend: Optional[AudioLib] = None) -> AudioData:
        """Alias of :meth:`~AudioReader.of`"""
        return cls.of(path, channel_first, backend)
