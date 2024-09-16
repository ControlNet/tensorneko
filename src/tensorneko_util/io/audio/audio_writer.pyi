from typing import overload, Union
from pathlib import Path

from .audio_data import AudioData
from ...backend.audio_lib import AudioLib
from ...util.type import T_ARRAY


class AudioWriter:

    @classmethod
    @overload
    def to(cls, path: Union[str, Path], audio: AudioData, channel_first: bool = True, backend: AudioLib = None
    ) -> None: ...

    @classmethod
    @overload
    def to(cls, path: Union[str, Path], audio: T_ARRAY, sample_rate: int = 16000, channel_first: bool = True,
        backend: AudioLib = None
    ): ...

    @overload
    def __new__(cls, path: Union[str, Path], audio: AudioData, channel_first: bool = True, backend: AudioLib = None
    ) -> None: ...

    @overload
    def __new__(cls, path: Union[str, Path], audio: T_ARRAY, sample_rate: int = 16000, channel_first: bool = True,
        backend: AudioLib = None
    ): ...
