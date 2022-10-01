from abc import ABC, abstractmethod


class AbstractFaceDetector(ABC):

    @abstractmethod
    def detect_face(self, img):
        pass

    @abstractmethod
    def detect_face_batch(self, imgs):
        pass
