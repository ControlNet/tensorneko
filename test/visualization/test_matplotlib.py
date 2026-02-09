import unittest

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorneko_util.visualization.multi_plots import MultiPlots
from tensorneko_util.visualization.matplotlib import plot_image


class TestMatplotlib(unittest.TestCase):
    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

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

    def test_plot_image_basic(self):
        """Test plot_image with a (H, W, C) ndarray."""
        image = np.random.rand(16, 16, 3).astype(np.float32)
        fig = plot_image(image)
        self.assertIsNotNone(fig)
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)

    def test_plot_image_with_title(self):
        image = np.random.rand(8, 8, 3).astype(np.float32)
        fig = plot_image(image, title="Test Title")
        ax = fig.get_axes()[0]
        self.assertEqual(ax.get_title(), "Test Title")

    def test_plot_image_with_figure(self):
        """Test passing an existing figure."""
        existing_fig = plt.figure()
        image = np.random.rand(8, 8, 3).astype(np.float32)
        fig = plot_image(image, figure=existing_fig)
        self.assertIs(fig, existing_fig)

    def test_plot_image_grayscale(self):
        """Test with a grayscale (H, W) image."""
        image = np.random.rand(16, 16).astype(np.float32)
        fig = plot_image(image, cmap="gray")
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.get_axes()), 1)
