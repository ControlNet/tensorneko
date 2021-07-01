from typing import Union

import torchaudio.backend.sox_io_backend
from torchaudio.backend.common import AudioMetaData

from .audio_data import AudioData


class AudioReader:
    """AudioReader for reading audio file"""

    @staticmethod
    def of(path: str, return_info: bool = False) -> Union[AudioData, AudioMetaData]:
        """
        Read audio tensor from given file.

        Args:
            path (``str``): Path to the audio file.
            return_info (``bool``, optional): True will return info rather than audio itself. Default ``False``.

        Returns:
            :class:`~.audio_data.AudioData` | :class:`~torchaudio.backend.common.AudioMetaData`:
                The Audio data in the file.
        """
        if return_info:
            return torchaudio.info(path)
        else:
            return AudioData(*torchaudio.load(path))
