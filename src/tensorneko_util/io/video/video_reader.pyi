from typing import Optional, overload, Union
from pathlib import Path

from numpy import ndarray

from .video_data import VideoData
from ...backend.visual_lib import VisualLib


class VideoReader:

    @classmethod
    def of(cls, path: Union[str, Path], channel_first: bool = True, backend: Optional[VisualLib] = None) -> VideoData:
        ...

    @classmethod
    def with_indexes(cls, path: Union[str, Path], indexes: ndarray,
        channel_first: bool = True, backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...

    @classmethod
    @overload
    def with_range(cls, path: Union[str, Path], start: int, end: int, step: int, channel_first: bool = True,
        backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...

    @classmethod
    @overload
    def with_range(cls, path: Union[str, Path], end: int, channel_first: bool = True,
        backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...

    def __new__(cls, path: Union[str, Path], channel_first: bool = True, backend: Optional[VisualLib] = None
    ) -> VideoData:
        ...
