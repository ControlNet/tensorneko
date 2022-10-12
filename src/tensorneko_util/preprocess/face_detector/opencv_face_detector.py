from typing import Union, Optional, Tuple

import numpy as np
from numpy import ndarray

from .abstract_face_detector import AbstractFaceDetector
from ..crop import crop_with_padding
from ..image import rgb2gray
from ...backend import VisualLib


class OpencvFaceDetector(AbstractFaceDetector):
    """
    The face detector implemented in OpenCV.
    The library OpenCV and opencv-contrib-python are required.

    Examples::

        from tensorneko_util.preprocess.face_detector import FaceXZooFaceDetector

        detector = OpencvFaceDetector()

        # process image, save to opencv_face_detector_out.jpg
        detector.process_image("test/resource/image_sample/1.183.jpg",
            "test/resource/temp_output/opencv_face_detector_out.jpg")

        # process video, save to opencv_face_detector_out.mp4
        detector.process_video("test/resource/video_sample/sample.mp4",
            "test/resource/temp_output/opencv_face_detector_out.mp4")

    """

    def __init__(self):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV and opencv-contrib-python is not installed. Please install it first.")

        try:
            import cv2.data
        except ImportError:
            raise ImportError("opencv-contrib-python is not installed. Please install it first.")

        self.cv2 = cv2
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_face(self, image: ndarray, *args, **kwargs):
        # img should be gray scale
        if len(image.shape) == 3:
            image = rgb2gray((image * 255).astype(int), backend=VisualLib.OPENCV)

        faces = self.face_cascade.detectMultiScale(image, *args, **kwargs)
        return faces

    def process_image(self, image_path: str, out_path: str, max_faces=1, margin=0, *args, **kwargs):
        if max_faces > 1:
            raise NotImplementedError("Multiple faces are not supported yet.")

        img = self.cv2.imread(image_path)
        gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        faces = self.detect_face(gray, *args, **kwargs)
        for (x, y, w, h) in faces[:max_faces]:
            cropped = crop_with_padding(img, x - margin, x + w + margin, y - margin, y + h + margin)
            self.cv2.imwrite(out_path, cropped)

    def _infer_frame_size(self, video_path: str, margin: int = 0) -> Tuple[int, int]:
        # get first face crop in the video
        video = self.cv2.VideoCapture(video_path)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            faces = self.detect_face(gray)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                side = max(w, h)
                video.release()
                return side + 2 * margin, side + 2 * margin

    def process_video(self, video_path: str, out_path: str, frame_size: Optional[Union[int, Tuple[int, int]]] = None,
        margin=0, fourcc="mp4v", *args, **kwargs
    ):
        video = self.cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError("Cannot open video file: " + video_path)

        # infer frame size
        if frame_size is None:
            frame_size = self._infer_frame_size(video_path, margin)

        if type(frame_size) is int:
            frame_size = frame_size, frame_size

        fps = video.get(self.cv2.CAP_PROP_FPS)
        writer = self.cv2.VideoWriter(out_path, self.cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
        x1, y1, x2, y2 = 0, 0, 1, 1
        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            faces = self.detect_face(gray, *args, **kwargs)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                side = max(w, h)
                center = (x + w // 2, y + h // 2)
                x1, y1 = center[0] - side // 2, center[1] - side // 2
                x2, y2 = center[0] + side // 2, center[1] + side // 2

            cropped = crop_with_padding(frame, x1 - margin, x2 + margin, y1 - margin, y2 + margin)
            resized = self.cv2.resize(cropped, frame_size)
            writer.write(resized)
        video.release()
        writer.release()
