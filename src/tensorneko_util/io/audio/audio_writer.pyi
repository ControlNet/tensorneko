from typing import overload

from .audio_data import AudioData
from ...backend.audio_lib import AudioLib
from ...util.type import T_ARRAY


class AudioWriter:

    @classmethod
    @overload
    def to(cls, path: str, audio: AudioData, backend: AudioLib = None) -> None: ...

    @classmethod
    @overload
    def to(cls, path: str, audio: T_ARRAY, sample_rate: int = 16000, backend: AudioLib = None): ...

    @overload
    def __new__(cls, path: str, audio: AudioData, backend: AudioLib = None) -> None: ...

    @overload
    def __new__(cls, path: str, audio: T_ARRAY, sample_rate: int = 16000, backend: AudioLib = None): ...
