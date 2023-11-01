import unittest
import torch

from tensorneko.dataset.sampler import SequentialIterSampler
from torch.utils.data import TensorDataset, DataLoader

class TestSequentialIterSampler(unittest.TestCase):

    def test_sequential_iter_sampler(self):
        dataset = TensorDataset(torch.arange(10))
        data_loader = DataLoader(dataset, batch_size=2, sampler=SequentialIterSampler(dataset, 4), drop_last=False)
        self.assertTrue([each[0].tolist() for each in data_loader], [[0, 1], [2, 3]])
        self.assertTrue([each[0].tolist() for each in data_loader], [[4, 5], [6, 7]])
        self.assertTrue([each[0].tolist() for each in data_loader], [[8, 9], [0, 1]])
