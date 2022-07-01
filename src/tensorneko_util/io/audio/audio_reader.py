from typing import Union

from .audio_data import AudioData

try:
    import torchaudio
    from torchaudio.backend.common import AudioMetaData
except ImportError:
    pass


class AudioReader:
    """AudioReader for reading audio file"""

    @staticmethod
    def of(path: str, return_info: bool = False) -> Union[AudioData, "AudioMetaData"]:
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

    def __new__(cls, path: str, return_info: bool = False) -> Union[AudioData, "AudioMetaData"]:
        """Alias of :meth:`~AudioReader.of`"""
        return cls.of(path, return_info)
