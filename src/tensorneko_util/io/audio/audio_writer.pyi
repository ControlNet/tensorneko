from typing import overload, Union

from numpy import ndarray
from torch import Tensor

from ...io.audio import AudioData


class AudioWriter:

    @staticmethod
    @overload
    def to(path: str, audio: AudioData) -> None: ...

    @staticmethod
    @overload
    def to(path: str, audio: Union[Tensor, ndarray], sample_rate: int = 16000) -> None: ...
