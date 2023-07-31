from enum import Enum
from typing import Optional

from .visual_lib import VisualLib


class AudioLib(Enum):
    PYTORCH = 1
    FFMPEG = 2

    _is_torchaudio_available: Optional[bool] = None

    @classmethod
    def pytorch_available(cls) -> bool:
        if cls._is_torchaudio_available is None:
            try:
                import torchaudio
                cls._is_torchaudio_available = True
            except ImportError:
                cls._is_torchaudio_available = False
        return cls._is_torchaudio_available

    @classmethod
    def ffmpeg_available(cls) -> bool:
        return VisualLib.ffmpeg_available()
