from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional, Union

from ..crop import crop_with_padding

if TYPE_CHECKING:
    from .abstract_face_detector import AbstractFaceDetector


def detector_model_crop_image(detector: AbstractFaceDetector, image_path: str, out_path: str, max_faces=1, margin=0):
    if max_faces > 1:
        raise NotImplementedError("Multiple faces are not supported yet.")
    import cv2

    frame = cv2.imread(image_path)
    dets = detector.detect_face(frame)
    for det in dets[:max_faces]:
        x1, y1, x2, y2, _ = det
        cropped = crop_with_padding(frame, int(x1 - margin), int(x2 + margin), int(y1 - margin), int(y2 + margin))
        cv2.imwrite(out_path, cropped)


def _infer_frame_size(detector: AbstractFaceDetector, video_path: str, margin: int = 0
) -> Tuple[int, int]:
    import cv2
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        dets = detector.detect_face(frame)
        if len(dets) > 0:
            x1, y1, x2, y2, confidence = dets[0]
            # center
            side = int(max(abs(x2 - x1), abs(y2 - y1)))
            video.release()
            return side + 2 * margin, side + 2 * margin


def detector_model_crop_video(detector: AbstractFaceDetector, video_path: str, out_path: str,
    frame_size: Optional[Union[int, Tuple[int, int]]] = None, margin=0, fourcc="mp4v"
):
    import cv2
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Cannot open video file: " + video_path)

    # infer frame size
    if frame_size is None:
        frame_size = _infer_frame_size(detector, video_path, margin)

    if type(frame_size) is int:
        frame_size = frame_size, frame_size

    fps = video.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
    x1, y1, x2, y2 = 0, 0, 1, 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
        dets = detector.detect_face(frame)
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
        resized = cv2.resize(cropped, frame_size)
        writer.write(resized)
    video.release()
    writer.release()
