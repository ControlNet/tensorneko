import unittest
from unittest.mock import patch

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from tensorneko_util.visualization.multi_plots import (
    MultiPlots,
    ImagePlot,
    PlotPlot,
    PlotType,
)


class TestMultiPlots(unittest.TestCase):
    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_add_image_and_plot(self):
        mp = MultiPlots(n_row=1, n_col=2)
        img = np.random.rand(10, 10, 3)
        mp.add_image(img, title="Test Image")
        self.assertEqual(len(mp.subplots), 1)
        self.assertIsInstance(mp.subplots[0], ImagePlot)
        self.assertEqual(mp.subplots[0].plot_type, PlotType.IMAGE)

        mp.add_plot(lambda ax: ax)
        self.assertEqual(len(mp.subplots), 2)
        self.assertIsInstance(mp.subplots[1], PlotPlot)
        self.assertEqual(mp.subplots[1].plot_type, PlotType.PLOT)

    def test_plot_renders_images(self):
        mp = MultiPlots(n_row=1, n_col=2)
        img1 = np.random.rand(8, 8, 3)
        img2 = np.random.rand(8, 8, 3)
        mp.add_image(img1, title="Img1")
        mp.add_image(img2, title="Img2")
        fig = mp.plot()
        self.assertIsNotNone(fig)
        axes = fig.get_axes()
        self.assertEqual(len(axes), 2)

    def test_plot_renders_plot_func(self):
        def my_plot(ax):
            ax.plot([1, 2, 3], [4, 5, 6])
            return ax

        mp = MultiPlots(n_row=1, n_col=1)
        mp.add_plot(my_plot)
        fig = mp.plot()
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)

    def test_determine_shape_auto_row(self):
        mp = MultiPlots(add_by_row=True)
        mp.add_image(np.zeros((4, 4, 3)))
        mp.add_image(np.zeros((4, 4, 3)))
        mp.add_image(np.zeros((4, 4, 3)))
        mp._determine_shape()
        self.assertEqual(mp.n_row, 1)
        self.assertEqual(mp.n_col, 3)

    def test_determine_shape_auto_col(self):
        mp = MultiPlots(add_by_row=False)
        mp.add_image(np.zeros((4, 4, 3)))
        mp.add_image(np.zeros((4, 4, 3)))
        mp._determine_shape()
        self.assertEqual(mp.n_row, 2)
        self.assertEqual(mp.n_col, 1)

    def test_determine_shape_n_col_only(self):
        mp = MultiPlots(n_col=2)
        for _ in range(5):
            mp.add_image(np.zeros((4, 4, 3)))
        mp._determine_shape()
        self.assertEqual(mp.n_row, 3)  # ceil(5/2)
        self.assertEqual(mp.n_col, 2)

    def test_determine_shape_n_row_only(self):
        mp = MultiPlots(n_row=2)
        for _ in range(5):
            mp.add_image(np.zeros((4, 4, 3)))
        mp._determine_shape()
        self.assertEqual(mp.n_row, 2)
        self.assertEqual(mp.n_col, 3)  # ceil(5/2)

    def test_determine_shape_insufficient_raises(self):
        mp = MultiPlots(n_row=1, n_col=1)
        mp.add_image(np.zeros((4, 4, 3)))
        mp.add_image(np.zeros((4, 4, 3)))
        with self.assertRaises(ValueError):
            mp._determine_shape()

    def test_title(self):
        mp = MultiPlots(n_row=1, n_col=1, title="My Title")
        mp.add_image(np.zeros((4, 4, 3)))
        fig = mp.plot()
        self.assertEqual(fig._suptitle.get_text(), "My Title")

    def test_mixed_image_and_plot(self):
        mp = MultiPlots(n_row=1, n_col=3)
        mp.add_image(np.random.rand(4, 4, 3), title="Image")
        mp.add_plot(lambda ax: ax)
        mp.add_image(np.random.rand(4, 4, 3))
        fig = mp.plot()
        self.assertEqual(len(fig.get_axes()), 3)


if __name__ == "__main__":
    unittest.main()
