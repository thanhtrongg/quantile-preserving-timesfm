import unittest

import numpy as np

from src.evaluation.gift_eval import GiftEvalWindow, aggregate_diagnostics, forecast_windows


class FakeAdapter:
    def forecast(self, histories, prediction_length):
        batch = len(histories)
        levels = np.asarray([0.1, 0.5, 0.9])
        point = np.zeros((batch, prediction_length), dtype=np.float32)
        quantiles = np.zeros((batch, prediction_length, levels.size), dtype=np.float32)
        return {
            "point_forecast": point,
            "quantile_forecasts": quantiles,
            "quantile_levels": levels,
        }


class SmokePipelineTest(unittest.TestCase):
    def test_model_to_evaluation_smoke_without_benchmark_substitution(self):
        window = GiftEvalWindow(
            dataset="toy/1H/short",
            series_id="s0",
            window_id=0,
            history=np.arange(20, dtype=np.float32),
            ground_truth=np.asarray([20.0, 21.0, 22.0], dtype=np.float32),
            timestamps=("2026-01-01T20:00:00", "2026-01-01T21:00:00", "2026-01-01T22:00:00"),
            frequency="H",
        )
        results = forecast_windows(FakeAdapter(), [window], prediction_length=3)
        metrics = aggregate_diagnostics(results)
        self.assertEqual(metrics["num_windows"], 1)
        self.assertEqual(metrics["horizon"], 3)
        self.assertEqual(metrics["quantile_levels"], [0.1, 0.5, 0.9])
        # This test is only a pipeline contract smoke test; it is not a GIFT-Eval run.
        self.assertTrue(np.array_equal(window.history, np.arange(20, dtype=np.float32)))

