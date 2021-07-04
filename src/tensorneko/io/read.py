from .audio import AudioReader
from .image import ImageReader
from .text import TextReader
from .video import VideoReader


class read:
    """
    A read util for read different modality of data.

    Examples:

        Read audio:

        .. code-block:: python

            audio: AudioData = read.audio.of(path, return_info=False)

        Read video:

        .. code-block:: python

            video: VideoData = read.video.of(path)

        Read image:

        .. code-block:: python

            image: Tensor = read.image.of(path)

        Read text:

        .. code-block:: python

            text: str = read.text.of(path)

    """

    video = VideoReader
    audio = AudioReader
    text = TextReader
    image = ImageReader
