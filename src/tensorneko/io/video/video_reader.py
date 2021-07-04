from torchvision.io import read_video

from .video_data import VideoData


class VideoReader:
    """VideoReader for reading video file"""

    @staticmethod
    def of(path: str) -> VideoData:
        """
        Read video tensor from given file.

        Args:
            path: Path to the video file.

        Returns:
            :class:`~torch.Tensor`: A float tensor of video (T, C, H, W), with value range of 0. to 1.
        """
        video, audio, info = read_video(path)
        video = video.permute(0, 3, 1, 2) / 255
        return VideoData(video, audio, info)
