from unittest import TestCase

import numpy as np

from tensorneko.util import AverageMeter


class UtilAverageMeterTest(TestCase):

    items = np.random.rand(10)

    def test_average_meter(self):
        am = AverageMeter()
        for item in self.items:
            am.update(item)

        self.assertEqual(np.round(am.avg, 3), np.round(np.mean(self.items), 3))
        self.assertEqual(np.round(am.sum, 3), np.round(np.sum(self.items), 3))
        self.assertEqual(am.count, len(self.items))
        self.assertEqual(am.val, self.items[-1])
