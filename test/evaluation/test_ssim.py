import unittest

import torch

from tensorneko.evaluation import ssim_image, ssim_video


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

        ssim = ssim_image(x, y, reduction="none")
        self.assertTrue(ssim.shape == torch.Size([10]))


if __name__ == '__main__':
    unittest.main()
