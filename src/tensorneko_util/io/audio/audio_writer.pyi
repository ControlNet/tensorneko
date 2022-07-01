from typing import overload

from .audio_data import AudioData
from ...util.type import T_ARRAY


class AudioWriter:

    @staticmethod
    @overload
    def to(path: str, audio: AudioData) -> None: ...

    @staticmethod
    @overload
    def to(path: str, audio: T_ARRAY, sample_rate: int = 16000) -> None: ...
