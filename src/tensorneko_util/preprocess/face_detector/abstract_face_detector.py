import glob
import os.path
from abc import ABC, abstractmethod

from numpy import ndarray


class AbstractFaceDetector(ABC):

    @abstractmethod
    def detect_face(self, image: ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def process_image(self, image_path: str, out_path: str, *args, **kwargs):
        pass

    @abstractmethod
    def process_video(self, video_path: str, out_path: str, *args, **kwargs):
        pass

    def process_image_dir(self, image_dir: str, out_dir: str, pattern="*.jpg", *args, **kwargs):
        all_images = glob.glob(os.path.join(image_dir, pattern), root_dir=image_dir)
        for image_path in all_images:
            out_path = os.path.join(out_dir, image_path)
            self.process_image(image_path, out_path, *args, **kwargs)

    def process_video_dir(self, video_dir: str, out_dir: str, pattern="*.mp4", *args, **kwargs):
        all_videos = glob.glob(os.path.join(video_dir, pattern), root_dir=video_dir)
        for video_path in all_videos:
            out_path = os.path.join(out_dir, video_path)
            self.process_video(video_path, out_path, *args, **kwargs)
