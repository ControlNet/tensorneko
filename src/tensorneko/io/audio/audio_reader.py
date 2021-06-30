from typing import Union

import torchaudio.backend.sox_io_backend
from torchaudio.backend.common import AudioMetaData

from .audio_data import AudioData


class AudioReader:
    @staticmethod
    def of(path: str, return_info: bool) -> Union[AudioData, AudioMetaData]:
        if return_info:
            return torchaudio.info(path)
        else:
            return AudioData(*torchaudio.load(path))
