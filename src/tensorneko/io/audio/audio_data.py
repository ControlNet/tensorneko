from dataclasses import dataclass

from torch import Tensor


@dataclass
class AudioData:
    audio: Tensor
    sample_rate: int