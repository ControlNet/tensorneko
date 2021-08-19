from typing import Union, overload

import torchaudio
from numpy import ndarray
from torch import Tensor

from tensorneko.io.audio import AudioData


class AudioWriter:

    @staticmethod
    @overload
    def to(path: str, audio: AudioData):
        pass

    @staticmethod
    @overload
    def to(path: str, audio: Union[Tensor, ndarray], sample_rate: int = 16000):
        pass

    @staticmethod
    def to(path: str, audio: Union[Tensor, ndarray, AudioData], sample_rate: int = 16000):
        if type(audio) == AudioData:
            torchaudio.save(path, audio.audio, audio.sample_rate)
        else:
            torchaudio.save(path, audio, sample_rate)
