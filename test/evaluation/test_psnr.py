import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from tensorneko.evaluation import psnr_image, psnr_video
from tensorneko.evaluation.enum import Reduction
from tensorneko.io import write
from tensorneko_util.backend.visual_lib import VisualLib
from tensorneko_util.io.image.image_reader import ImageReader


class TestPsnr(unittest.TestCase):
    def test_psnr_image_from_tensor_single(self):
        x = torch.rand(3, 256, 256)
        y = torch.rand(3, 256, 256)

        psnr = psnr_image(x, y)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_image_from_tensor_batch_reduce_mean(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        psnr = psnr_image(x, y)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_image_from_tensor_batch_reduce_none(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        psnr = psnr_image(x, y, reduction=Reduction.NONE)
        self.assertTrue(psnr.shape == torch.Size([10]))

    def test_psnr_video_from_tensor(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        psnr = psnr_video(x, y)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_image_from_tensor_batch_reduce_sum(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        psnr = psnr_image(x, y, reduction=Reduction.SUM)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_image_from_uint8_tensor(self):
        x = torch.randint(0, 256, (10, 3, 256, 256), dtype=torch.uint8)
        y = torch.randint(0, 256, (10, 3, 256, 256), dtype=torch.uint8)

        psnr = psnr_image(x, y)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_image_3d_input(self):
        x = torch.rand(3, 32, 32)
        y = torch.rand(3, 32, 32)

        psnr = psnr_image(x, y)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_image_from_path(self):
        tmpdir = tempfile.mkdtemp()
        pred_path = os.path.join(tmpdir, "pred.png")
        real_path = os.path.join(tmpdir, "real.png")
        pred_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        real_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(pred_np).save(pred_path)
        Image.fromarray(real_np).save(real_path)

        import tensorneko.evaluation.psnr as psnr_module
        from torchmetrics.functional import peak_signal_noise_ratio as orig_psnr

        class _PytorchImageReader:
            def __new__(cls, path, channel_first=True, backend=None):
                return ImageReader.of(
                    path, channel_first=channel_first, backend=VisualLib.PYTORCH
                )

        def _fixed_psnr(
            preds,
            target,
            data_range=None,
            reduction="elementwise_mean",
            dim=None,
            base=10.0,
        ):
            if reduction == "mean":
                reduction = "elementwise_mean"
            return orig_psnr(
                preds,
                target,
                data_range=data_range,
                reduction=reduction,
                dim=dim,
                base=base,
            )

        old_image = psnr_module.read.image
        old_psnr = psnr_module.psnr
        psnr_module.read.image = _PytorchImageReader
        psnr_module.psnr = _fixed_psnr
        try:
            result = psnr_image(pred_path, real_path)
            self.assertTrue(result.shape == torch.Size([]))
        finally:
            psnr_module.read.image = old_image
            psnr_module.psnr = old_psnr

    def test_psnr_video_from_path(self):
        tmpdir = tempfile.mkdtemp()
        pred_path = os.path.join(tmpdir, "pred.mp4")
        real_path = os.path.join(tmpdir, "real.mp4")
        pred_vid = torch.rand(4, 3, 32, 32)
        real_vid = torch.rand(4, 3, 32, 32)
        write.video(pred_path, pred_vid, 24.0, channel_first=True)
        write.video(real_path, real_vid, 24.0, channel_first=True)
        psnr = psnr_video(pred_path, real_path)
        self.assertTrue(psnr.shape == torch.Size([]))

    def test_psnr_video_from_uint8_tensor(self):
        x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        y = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        psnr = psnr_video(x, y)
        self.assertTrue(psnr.shape == torch.Size([]))


if __name__ == "__main__":
    unittest.main()
