from typing import Union

from .audio_data import AudioData
from .._default_backends import _default_audio_io_backend
from ...backend.audio_lib import AudioLib
from ...util import dispatch
from ...util.type import T_ARRAY


class AudioWriter:
    """AudioWriter for writing audio file"""

    @classmethod
    @dispatch
    def to(cls, path: str, audio: AudioData, channel_first: bool = True, backend: AudioLib = None) -> None:
        """
        Save wav file from :class:`~tensorneko.io.audio.audio_data.AudioData`.

        Args:
            path (``str``): The path of output file.
            audio (:class:`~tensorneko.io.audio.AudioData`): The AudioData object for output.
            channel_first (``bool``, optional): Whether the audio is channel first. The input shape is (C, T) if true
                and (T, C) if false. Default: True.
            backend (:class:`~tensorneko.io.audio.audio_lib.AudioLib`, optional): The audio library to use.
                Default: pytorch.
        """
        return cls.to(path, audio.audio, audio.sample_rate, channel_first, backend)

    @classmethod
    @dispatch
    def to(cls, path: str, audio: T_ARRAY, sample_rate: int = 16000, channel_first: bool = True,
        backend: AudioLib = None
    ):
        """
        Save wav file from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, T).

        Args:
            path (``str``): The path of output file.
            audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The tensor or array of audio.
            sample_rate (``int``, optional): The sample rate of the audio. Default: 16000.
            channel_first (``bool``, optional): Whether the audio is channel first. The input shape is (C, T) if true
                and (T, C) if false. Default: True.
            backend (:class:`~tensorneko.io.audio.audio_lib.AudioLib`, optional): The audio library to use.
                Default: pytorch.
        """
        backend = backend or _default_audio_io_backend()

        if backend == AudioLib.PYTORCH:
            if not AudioLib.pytorch_available():
                raise ValueError("Torchaudio is not available.")
            import torchaudio
            torchaudio.save(path, audio, sample_rate, channels_first=channel_first)
        else:
            raise ValueError("Unknown audio library: {}".format(backend))

    def __new__(cls, path: str, audio: Union[AudioData, T_ARRAY], *args, **kwargs):
        """Alias to :meth:`~AudioWriter.to`"""
        return cls.to(path, audio, *args, **kwargs)
