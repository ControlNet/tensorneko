from typing import Optional, overload

from numpy import ndarray

from .video_data import VideoData
from ...backend.visual_lib import VisualLib


class VideoReader:

    @classmethod
    def of(cls, path: str, channel_first: bool = True, backend: Optional[VisualLib] = None) -> VideoData:
        ...

    @classmethod
    def with_indexes(cls, path: str, indexes: ndarray,
        channel_first: bool = True, backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...

    @classmethod
    @overload
    def with_range(cls, path: str, start: int, end: int, step: int, channel_first: bool = True,
        backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...

    @classmethod
    @overload
    def with_range(cls, path: str, end: int, channel_first: bool = True,
        backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...

    def __new__(cls, path: str, channel_first: bool = True, backend: Optional[VisualLib] = None) -> VideoData:
        ...
