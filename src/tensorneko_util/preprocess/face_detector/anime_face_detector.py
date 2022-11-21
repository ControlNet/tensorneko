from typing import List, Optional, Union, Tuple

from numpy import ndarray

from ._utils import detector_model_crop_image, detector_model_crop_video
from .abstract_face_detector import AbstractFaceDetector


class AnimeFaceDetector(AbstractFaceDetector):
    """
    hysts/anime-face-detector
    https://github.com/hysts/anime-face-detector
    """

    def __init__(self, device: str = "cuda:0", box_scale_factor: float = 1.1):
        super().__init__()
        try:
            from anime_face_detector import create_detector
            import cv2
        except ImportError:
            raise ImportError("anime-face-detector is not installed. Please install it first. Please check the "
                              "details in https://github.com/hysts/anime-face-detector")
        else:
            self.cv2 = cv2
            self.detector = create_detector(device=device, box_scale_factor=box_scale_factor)

    def detect_face(self, image: ndarray, *args, **kwargs) -> List[ndarray]:
        assert image.ndim == 3 and image.shape[2] == 3, "frame should be 3-dim"
        return [each["bbox"] for each in self.detector.detect(image)]

    def crop_image(self, image_path: str, out_path: str, max_faces=1, margin=0, *args, **kwargs) -> None:
        detector_model_crop_image(self, image_path, out_path, max_faces, margin)

    def crop_video(self, video_path: str, out_path: str, frame_size: Optional[Union[int, Tuple[int, int]]] = None,
        margin=0, fourcc="mp4v", *args, **kwargs
    ) -> None:
        detector_model_crop_video(self, video_path, out_path, frame_size, margin, fourcc)

