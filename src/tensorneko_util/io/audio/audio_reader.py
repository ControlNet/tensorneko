from typing import Optional

from .audio_data import AudioData
from .._default_backends import _default_audio_io_backend
from ...backend.audio_lib import AudioLib


class AudioReader:
    """AudioReader for reading audio file"""

    @staticmethod
    def of(path: str, backend: Optional[AudioLib] = None) -> AudioData:
        """
        Read audio tensor from given file.

        Args:
            path (``str``): Path to the audio file.
            backend (:class:`~tensorneko.io.audio.audio_lib.AudioLib`, optional): The audio library to use.
                Default: pytorch.

        Returns:
            :class:`~.audio_data.AudioData` | :class:`~torchaudio.backend.common.AudioMetaData`:
                The Audio data in the file.
        """
        backend = backend or _default_audio_io_backend()

        if backend == AudioLib.PYTORCH:
            if not AudioLib.pytorch_available():
                raise ValueError("Torchaudio is not available.")
            import torchaudio
            return AudioData(*torchaudio.load(path))
        else:
            raise ValueError("Unknown audio library: {}".format(backend))

    def __new__(cls, path: str, backend: Optional[AudioLib] = None) -> AudioData:
        """Alias of :meth:`~AudioReader.of`"""
        return cls.of(path, backend)
