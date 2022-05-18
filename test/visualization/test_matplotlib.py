import unittest

import numpy as np

from tensorneko_util.visualization.multi_plots import MultiPlots


class TestMatplotlib(unittest.TestCase):

    def test_multiplots_location_by_row(self):
        multi_plots = MultiPlots(n_row=3, n_col=5, add_by_row=True)
        inputs = np.arange(15).reshape(3, 5)

        calculated = np.vectorize(multi_plots._location)(inputs)
        expected = np.arange(1, 16).reshape(3, 5)
        np.testing.assert_array_equal(calculated, expected)

    def test_multiplots_location_by_column(self):
        multi_plots = MultiPlots(n_row=3, n_col=5, add_by_row=False)
        inputs = np.arange(15).reshape(3, 5)

        calculated = np.vectorize(multi_plots._location)(inputs)
        expected = np.arange(1, 16).reshape(5, 3).T
        np.testing.assert_array_equal(calculated, expected)
