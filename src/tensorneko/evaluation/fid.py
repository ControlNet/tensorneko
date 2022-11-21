from __future__ import annotations

from dataclasses import dataclass
from typing import Union, List

import torch
import torchvision.io
from torch import Tensor
from torch.types import Device
from torch.utils.data import IterableDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import resize

from tensorneko_util.backend import VisualLib
from tensorneko_util.backend._tqdm import import_tqdm_auto

try:
    from typing import Literal
    TypeOption = Literal["video", "image"]
except ImportError:
    TypeOption = str


class FID:

    def __init__(self, device: Union[str, Device] = "cpu"):
        self.device = torch.device(device)
        self.fid = FrechetInceptionDistance().to(self.device)
        self.pred_data = _FIDDataset()
        self.true_data = _FIDDataset()

    def add_pred_image(self, path: str) -> FID:
        self.pred_data.add_image(path)
        return self

    def add_pred_video(self, path: str) -> FID:
        self.pred_data.add_video(path)
        return self

    def add_true_image(self, path: str) -> FID:
        self.true_data.add_image(path)
        return self

    def add_true_video(self, path: str) -> FID:
        self.true_data.add_video(path)
        return self

    def to(self, device: Union[str, Device]) -> FID:
        self.device = torch.device(device)
        self.fid.to(self.device)
        return self

    def cpu(self) -> FID:
        return self.to("cpu")

    def cuda(self) -> FID:
        return self.to("cuda")

    def compute(self, batch_size=128, num_workers=8, progress_bar: bool = True) -> float:
        pred = torch.utils.data.DataLoader(self.pred_data, batch_size=batch_size, num_workers=num_workers)
        true = torch.utils.data.DataLoader(self.true_data, batch_size=batch_size, num_workers=num_workers)

        if progress_bar:
            tqdm = import_tqdm_auto().tqdm
            pred = tqdm(pred, desc="Forward predicted features")
            true = tqdm(true, desc="Forward ground truth features")

        for batch in pred:
            self.fid.update(batch.to(self.device), real=False)
        for batch in true:
            self.fid.update(batch.to(self.device), real=True)

        return self.fid.compute().item()


@dataclass
class _FIDEntry:
    type: TypeOption
    path: str


class _FIDDataset(IterableDataset):

    def __init__(self):
        super().__init__()
        self.content: List[_FIDEntry] = []
        self.cv2 = None
        self.length = 0

    def add_image(self, path: str):
        self.content.append(_FIDEntry("image", path))
        self.length += 1

    def add_video(self, path: str):
        if self.cv2 is None:
            if not VisualLib.opencv_available():
                raise RuntimeError("OpenCV is required to add video. Please install it first.")
            else:
                import cv2
                self.cv2 = cv2
        self.content.append(_FIDEntry("video", path))
        cap = self.cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file.")
        n_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        self.length += n_frames

    @staticmethod
    def _preprocess_image(image: Tensor) -> Tensor:
        image = resize(image, [299, 299])
        return image

    def _read_image(self, path: str) -> torch.Tensor:
        image = torchvision.io.read_image(path)
        image = self._preprocess_image(image)
        return image

    def _read_video(self, path: str) -> torch.Tensor:
        cap = self.cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            frame = self._preprocess_image(frame)
            yield frame

    def __iter__(self):
        for entry in self.content:
            if entry.type == "image":
                yield self._read_image(entry.path)
            elif entry.type == "video":
                yield from self._read_video(entry.path)
            else:
                raise RuntimeError("Unknown entry type.")

    def __len__(self):
        return self.length
