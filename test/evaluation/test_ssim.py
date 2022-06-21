import unittest

import torch

from tensorneko.evaluation import ssim_image, ssim_video
from tensorneko.evaluation.enum import Reduction


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


if __name__ == '__main__':
    unittest.main()
