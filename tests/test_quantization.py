import os
import unittest

import numpy as np
import pandas as pd

from src.metrics.quantization import (
    assert_context_alignment,
    assert_forecast_alignment,
    quantile_distortion,
)


def _frame(point_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for window_id in [0, 1]:
        for step in range(3):
            target = float(step + window_id)
            rows.append(
                {
                    "dataset": "toy/short",
                    "series_id": "s0",
                    "forecast_window_id": window_id,
                    "time_step": step,
                    "ground_truth": target,
                    "point_forecast": target + point_shift,
                    "q_0.1": target - 1.0 + point_shift,
                    "q_0.5": target + point_shift,
                    "q_0.9": target + 1.0 + point_shift,
                }
            )
    return pd.DataFrame(rows)


class QuantizationDiagnosticsTest(unittest.TestCase):
    def test_alignment_and_distortion(self):
        reference = _frame()
        candidate = _frame(0.25)
        contexts = pd.DataFrame(
            {
                "dataset": ["toy/short", "toy/short"],
                "series_id": ["s0", "s0"],
                "forecast_window_id": [0, 1],
                "history_values": [[1.0, 2.0], [2.0, 3.0]],
            }
        )
        assert_forecast_alignment(reference, candidate)
        assert_context_alignment(contexts, contexts.copy())
        distortion = quantile_distortion(
            reference, candidate, dataset="toy/short", precision="int8"
        )
        self.assertAlmostEqual(
            distortion.loc[distortion["quantile"] == 0.5, "mean_absolute_quantile_deviation"].iloc[0],
            0.25,
        )

    @unittest.skipUnless(
        os.getenv("RUN_QUANTIZATION_INTEGRATION") == "1",
        "Set RUN_QUANTIZATION_INTEGRATION=1 for the real checkpoint integration test",
    )
    def test_real_timesfm_int8_load_and_forecast(self):
        from src.models.timesfm import TimesFMAdapter, TimesFMConfig

        adapter = TimesFMAdapter(
            TimesFMConfig(
                dtype="float32",
                context_length=1024,
                max_horizon=128,
                torch_compile=False,
                quantization_backend="torchao_int8_weight_only",
            )
        )
        output = adapter.forecast([np.arange(1024, dtype=np.float32)], 30)
        self.assertEqual(output["point_forecast"].shape, (1, 30))
        self.assertEqual(output["quantile_forecasts"].shape, (1, 30, 9))
        self.assertTrue(np.isfinite(output["point_forecast"]).all())
        self.assertTrue(np.isfinite(output["quantile_forecasts"]).all())
