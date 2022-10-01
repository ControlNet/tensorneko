from numpy import ndarray

from .abstract_face_detector import AbstractFaceDetector
from ..image import rgb2gray
from ...backend import VisualLib


class OpencvFaceDetector(AbstractFaceDetector):

    def __init__(self):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is not installed. Please install it first.")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_face(self, img: ndarray, *args, **kwargs):
        # img should be gray scale
        if len(img.shape) == 3:
            img = rgb2gray((img * 255).astype(int), backend=VisualLib.OPENCV)

        faces = self.face_cascade.detectMultiScale(img, *args, **kwargs)
        return faces

    def detect_face_batch(self, imgs: ndarray, *args, **kwargs):
        return [self.detect_face(img, *args, **kwargs) for img in imgs]
