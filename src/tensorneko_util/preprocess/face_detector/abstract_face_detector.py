import glob
import os.path
from abc import ABC, abstractmethod

from numpy import ndarray

from ...backend._tqdm import import_tqdm_auto


class AbstractFaceDetector(ABC):

    @abstractmethod
    def detect_face(self, image: ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def crop_image(self, image_path: str, out_path: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def crop_video(self, video_path: str, out_path: str, *args, **kwargs) -> None:
        pass

    def crop_image_dir(self, image_dir: str, out_dir: str, pattern="*.jpg", progress_bar = False, *args, **kwargs
    ) -> None:
        all_images = glob.glob(os.path.join(image_dir, pattern), root_dir=image_dir)
        if progress_bar:
            all_images = import_tqdm_auto().tqdm(all_images)
        for image_path in all_images:
            out_path = os.path.join(out_dir, image_path)
            self.crop_image(image_path, out_path, *args, **kwargs)

    def crop_video_dir(self, video_dir: str, out_dir: str, pattern="*.mp4", progress_bar = False, *args, **kwargs
    ) -> None:
        all_videos = glob.glob(os.path.join(video_dir, pattern), root_dir=video_dir)
        if progress_bar:
            all_videos = import_tqdm_auto().tqdm(all_videos)
        for video_path in all_videos:
            out_path = os.path.join(out_dir, video_path)
            self.crop_video(video_path, out_path, *args, **kwargs)
