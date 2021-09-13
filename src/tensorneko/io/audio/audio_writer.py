from typing import Union, overload

import torchaudio
from numpy import ndarray
from torch import Tensor

from .audio_data import AudioData


class AudioWriter:
    """AudioWriter for writing audio file"""

    @staticmethod
    @overload
    def to(path: str, audio: AudioData) -> None:
        """
        Save wav file from :class:`~tensorneko.io.audio.audio_data.AudioData`.

        Args:
            path (``str``): The path of output file.
            audio (:class:`~tensorneko.io.audio.AudioData`): The AudioData object for output.
        """
        ...

    @staticmethod
    @overload
    def to(path: str, audio: Union[Tensor, ndarray], sample_rate: int = 16000):
        """
        Save wav file from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, T).

        Args:
            path (``str``): The path of output file.
            audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The tensor or array of audio with (C, T).
            sample_rate (``int``, optional): The sample rate of the audio. Default: 16000.
        """
        ...

    @staticmethod
    def to(path: str, audio: Union[Tensor, ndarray, AudioData], sample_rate: int = 16000):
        """
        The implementation of :meth:`~tensorneko.io.audio.audio_writer.AudioWriter.to`.
        """
        if type(audio) == AudioData:
            torchaudio.save(path, audio.audio, audio.sample_rate)
        else:
            torchaudio.save(path, audio, sample_rate)
