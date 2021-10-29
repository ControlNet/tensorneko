from typing import Union

import torchaudio
from numpy import ndarray
from torch import Tensor

from .audio_data import AudioData
from ...util import dispatch


class AudioWriter:
    """AudioWriter for writing audio file"""

    @staticmethod
    @dispatch
    def to(path: str, audio: AudioData) -> None:
        """
        Save wav file from :class:`~tensorneko.io.audio.audio_data.AudioData`.

        Args:
            path (``str``): The path of output file.
            audio (:class:`~tensorneko.io.audio.AudioData`): The AudioData object for output.
        """
        torchaudio.save(path, audio.audio, audio.sample_rate)

    @staticmethod
    @dispatch
    def to(path: str, audio: Union[Tensor, ndarray], sample_rate: int = 16000):
        """
        Save wav file from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (C, T).

        Args:
            path (``str``): The path of output file.
            audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The tensor or array of audio with (C, T).
            sample_rate (``int``, optional): The sample rate of the audio. Default: 16000.
        """
        torchaudio.save(path, audio, sample_rate)
