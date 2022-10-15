import os.path
from typing import Optional, Union, Tuple

from numpy import ndarray

from .abstract_face_detector import AbstractFaceDetector
from ..crop import crop_with_padding
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

        detector = FaceXZooFaceDetector("../../Forked/FaceX-Zoo", "cuda:0")

        # process image, save to facexzoo_face_detector_out.jpg
        detector.process_image("test/resource/image_sample/1.183.jpg",
            "test/resource/temp_output/facexzoo_face_detector_out.jpg")

        # process video, save to facexzoo_face_detector_out.mp4
        detector.process_video("test/resource/video_sample/sample.mp4",
            "test/resource/temp_output/facexzoo_face_detector_out.mp4")

    """

    def __init__(self, face_sdk_path: Optional[str] = None, device: str = "cuda:0"):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV and opencv-contrib-python is not installed. Please install it first.")

        self.cv2 = cv2

        if face_sdk_path is not None:
            import sys
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

    def detect_face(self, image: ndarray, *args, **kwargs):
        assert image.ndim == 3 and image.shape[2] == 3, "frame should be 3-dim"
        dets = self.faceDetModelHandler.inference_on_image(image)
        return dets

    def process_image(self, image_path: str, out_path: str, max_faces=1, margin=0, *args, **kwargs):
        if max_faces > 1:
            raise NotImplementedError("Multiple faces are not supported yet.")

        frame = self.cv2.imread(image_path)
        dets = self.detect_face(frame)
        for det in dets[:max_faces]:
            x1, y1, x2, y2, confidence = det
            cropped = crop_with_padding(frame, int(x1 - margin), int(x2 + margin), int(y1 - margin), int(y2 + margin))
            self.cv2.imwrite(out_path, cropped)

    def _infer_frame_size(self, video_path: str, margin: int = 0) -> Tuple[int, int]:
        video = self.cv2.VideoCapture(video_path)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            dets = self.detect_face(frame)
            if len(dets) > 0:
                x1, y1, x2, y2, confidence = dets[0]
                # center
                side = int(max(abs(x2 - x1), abs(y2 - y1)))
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
            dets = self.detect_face(frame)
            if len(dets) > 0:
                x1, y1, x2, y2, confidence = dets[0]
                # center
                x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                side = int(max(abs(x2 - x1), abs(y2 - y1)))
                x1 = x - side // 2
                x2 = x + side // 2
                y1 = y - side // 2
                y2 = y + side // 2

            cropped = crop_with_padding(frame, int(x1 - margin), int(x2 + margin), int(y1 - margin), int(y2 + margin))
            resized = self.cv2.resize(cropped, frame_size)
            writer.write(resized)
        video.release()
        writer.release()
