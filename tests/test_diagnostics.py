import unittest

import numpy as np

from src.metrics.diagnostics import (
    interval_coverage,
    interval_width,
    pinball_loss,
    quantile_crossing_rate,
)


class DiagnosticsTest(unittest.TestCase):
    def test_pinball_loss_toy_array(self):
        self.assertAlmostEqual(
            pinball_loss(np.asarray([1.0]), np.asarray([[0.0]]), [0.5]), 0.5
        )

    def test_interval_coverage(self):
        self.assertEqual(
            interval_coverage(np.asarray([1.0, 2.0]), np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])),
            1.0,
        )

    def test_interval_width(self):
        self.assertEqual(interval_width(np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])), 2.0)

    def test_crossing_rate(self):
        quantiles = np.asarray([[[2.0, 1.0], [1.0, 3.0]]])
        self.assertAlmostEqual(quantile_crossing_rate(quantiles), 0.5)

