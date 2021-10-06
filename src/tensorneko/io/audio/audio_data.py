from dataclasses import dataclass

from torch import Tensor


@dataclass
class AudioData:
    """
    Output audio data for :class:`~.audio_reader.AudioReader`.

    Attributes:
        audio (:class:`~torch.Tensor`): audio float tensor of (C, T)
        sample_rate (``int``): sample rate of the audio

    """
    audio: Tensor
    sample_rate: int

    def __iter__(self):
        return iter((self.audio, self.sample_rate))
