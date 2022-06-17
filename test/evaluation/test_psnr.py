import unittest

import torch

from tensorneko.evaluation import psnr_image, psnr_video


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

        psnr = psnr_image(x, y, reduction="none")
        self.assertTrue(psnr.shape == torch.Size([10]))


if __name__ == '__main__':
    unittest.main()
