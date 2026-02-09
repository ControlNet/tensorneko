import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from tensorneko.evaluation import ssim_image, ssim_video
from tensorneko.evaluation.enum import Reduction
from tensorneko.io import write
from tensorneko_util.backend.visual_lib import VisualLib
from tensorneko_util.io.image.image_reader import ImageReader


class TestSsim(unittest.TestCase):
    def test_ssim_image_from_tensor_single(self):
        x = torch.rand(3, 256, 256)
        y = torch.rand(3, 256, 256)

        ssim = ssim_image(x, y)
        self.assertTrue(ssim.shape == torch.Size([]))

    def test_ssim_image_from_tensor_batch_reduce_mean(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        ssim = ssim_image(x, y)
        self.assertTrue(ssim.shape == torch.Size([]))

    def test_ssim_image_from_tensor_batch_reduce_none(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        ssim = ssim_image(x, y, reduction=Reduction.NONE)
        self.assertTrue(ssim.shape == torch.Size([10]))

    def test_ssim_video_from_tensor(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        ssim = ssim_video(x, y)
        self.assertTrue(ssim.shape == torch.Size([]))

    def test_ssim_image_from_tensor_batch_reduce_sum(self):
        x = torch.rand(10, 3, 256, 256)
        y = torch.rand(10, 3, 256, 256)

        ssim = ssim_image(x, y, reduction=Reduction.SUM)
        self.assertTrue(ssim.shape == torch.Size([]))

    def test_ssim_image_from_uint8_tensor(self):
        x = torch.randint(0, 256, (10, 3, 256, 256), dtype=torch.uint8)
        y = torch.randint(0, 256, (10, 3, 256, 256), dtype=torch.uint8)

        ssim = ssim_image(x, y)
        self.assertTrue(ssim.shape == torch.Size([]))

    def test_ssim_image_3d_input(self):
        x = torch.rand(3, 32, 32)
        y = torch.rand(3, 32, 32)

        ssim = ssim_image(x, y)
        self.assertTrue(ssim.shape == torch.Size([]))

    def test_ssim_image_from_path(self):
        tmpdir = tempfile.mkdtemp()
        pred_path = os.path.join(tmpdir, "pred.png")
        real_path = os.path.join(tmpdir, "real.png")
        pred_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        real_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(pred_np).save(pred_path)
        Image.fromarray(real_np).save(real_path)

        import tensorneko.evaluation.ssim as ssim_module

        class _PytorchImageReader:
            def __new__(cls, path, channel_first=True, backend=None):
                return ImageReader.of(
                    path, channel_first=channel_first, backend=VisualLib.PYTORCH
                )

        old_image = ssim_module.read.image
        ssim_module.read.image = _PytorchImageReader
        try:
            result = ssim_image(pred_path, real_path)
            self.assertTrue(result.shape == torch.Size([]))
        finally:
            ssim_module.read.image = old_image

    def test_ssim_video_from_path(self):
        tmpdir = tempfile.mkdtemp()
        pred_path = os.path.join(tmpdir, "pred.mp4")
        real_path = os.path.join(tmpdir, "real.mp4")
        pred_vid = torch.rand(4, 3, 64, 64)
        real_vid = torch.rand(4, 3, 64, 64)
        write.video(pred_path, pred_vid, 24.0, channel_first=True)
        write.video(real_path, real_vid, 24.0, channel_first=True)

        import tensorneko.evaluation.ssim as ssim_module
        from tensorneko_util.io.video.video_data import VideoData

        original_video_reader = ssim_module.read.video

        class _ChannelFirstVideoReader:
            def __new__(cls, path, *args, **kwargs):
                vdata = original_video_reader(path, *args, **kwargs)
                vdata.video = vdata.video.permute(0, 3, 1, 2).contiguous()
                return vdata

        old_video = ssim_module.read.video
        ssim_module.read.video = _ChannelFirstVideoReader
        try:
            result = ssim_video(pred_path, real_path)
            self.assertTrue(result.shape == torch.Size([]))
        finally:
            ssim_module.read.video = old_video

    def test_ssim_video_from_uint8_tensor(self):
        x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        y = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        result = ssim_video(x, y)
        self.assertTrue(result.shape == torch.Size([]))


if __name__ == "__main__":
    unittest.main()
