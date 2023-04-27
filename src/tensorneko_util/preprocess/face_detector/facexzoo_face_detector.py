import os.path
import sys
from typing import Optional, Union, Tuple

from numpy import ndarray

from ._utils import detector_model_crop_image, detector_model_crop_video
from .abstract_face_detector import AbstractFaceDetector
from ...io import read


class FaceXZooFaceDetector(AbstractFaceDetector):
    """
    The face sdk is in https://github.com/ControlNet/FaceX-Zoo.
    The fork of https://github.com/JDAI-CV/FaceX-Zoo/tree/main/face_sdk.

    Args:
        face_sdk_path (``str``): The path to the face_sdk path of FaceXZoo.
        device (``str``, optional): The device to use, default is "cuda:0".

    Examples::

        from tensorneko_util.preprocess.face_detector import FaceXZooFaceDetector

        detector = FaceXZooFaceDetector(FaceXZooFaceDetector.install(), "cuda:0")

        # process image, save to facexzoo_face_detector_out.jpg
        detector.crop_image("test/resource/image_sample/1.183.jpg",
            "test/resource/temp_output/facexzoo_face_detector_out.jpg")

        # process video, save to facexzoo_face_detector_out.mp4
        detector.crop_video("test/resource/video_sample/sample.mp4",
            "test/resource/temp_output/facexzoo_face_detector_out.mp4")

    """

    def __init__(self, face_sdk_path: Optional[str] = None, device: str = "cuda:0"):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV and opencv-contrib-python is not installed. Please install it first.")

        self.cv2 = cv2

        if face_sdk_path is not None:
            sys.path.append(face_sdk_path)
        else:
            if os.path.exists("FaceX-Zoo"):
                face_sdk_path = "FaceX-Zoo"
                sys.path.append(face_sdk_path)
        try:
            from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
            from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
        except ImportError:
            raise ImportError("FaceX-Zoo cannot be imported, please specify the path to the face_sdk path of FaceXZoo"
                              " or put it in the working directory.")

        model_conf = read.yaml(os.path.join(face_sdk_path, "config", "model_conf.yaml"))
        model_path = os.path.join(face_sdk_path, 'models')
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]

        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        self.faceDetModelHandler = FaceDetModelHandler(model, device, cfg)

    @staticmethod
    def install(path: Optional[str] = None) -> str:
        """
        Install FaceX-Zoo by clone from GitHub.

        Args:
            path (``str``, optional): The path to install FaceX-Zoo, default is "./FaceX-Zoo".

        Returns:
            ``str``: The path to the installed FaceX-Zoo.

        """
        path = path or "FaceX-Zoo"
        if os.path.exists(path):
            return path

        os.system(f"git clone --depth=1 https://github.com/ControlNet/FaceX-Zoo {path or ''}")
        return path

    def detect_face(self, image: ndarray, *args, **kwargs):
        assert image.ndim == 3 and image.shape[2] == 3, "frame should be 3-dim"
        dets = self.faceDetModelHandler.inference_on_image(image)
        return dets

    def crop_image(self, image_path: str, out_path: str, max_faces=1, margin=0, *args, **kwargs) -> None:
        detector_model_crop_image(self, image_path, out_path, max_faces, margin)

    def crop_video(self, video_path: str, out_path: str, frame_size: Optional[Union[int, Tuple[int, int]]] = None,
        margin=0, fourcc="mp4v", *args, **kwargs
    ) -> None:
        detector_model_crop_video(self, video_path, out_path, frame_size, margin, fourcc)
