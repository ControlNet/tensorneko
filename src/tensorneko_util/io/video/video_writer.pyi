from typing import overload, Optional

from .video_data import VideoData
from ...backend.visual_lib import VisualLib
from ...util.type import T_ARRAY


class VideoWriter:

    @classmethod
    @overload
    def to(cls, path: str, video: VideoData, audio_codec: str = None, channel_first: bool = False,
        backend: VisualLib = None
    ) -> None: ...

    @classmethod
    @overload
    def to(cls, path: str, video: T_ARRAY, video_fps: float, audio: T_ARRAY = None,
        audio_fps: Optional[int] = None, audio_codec: str = None, channel_first: bool = False,
        backend: VisualLib = None
    ) -> None: ...

    @overload
    def __new__(cls, path: str, video: VideoData, audio_codec: str = None, channel_first: bool = False,
        backend: VisualLib = None
    ) -> None: ...

    @overload
    def __new__(cls, path: str, video: T_ARRAY, video_fps: float, audio: T_ARRAY = None,
        audio_fps: Optional[int] = None, audio_codec: str = None, channel_first: bool = False,
        backend: VisualLib = None
    ) -> None: ...
