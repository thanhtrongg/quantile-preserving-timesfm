import unittest

import numpy as np

from src.models.timesfm import TimesFMAdapter, TimesFMConfig, normalize_forecast_output


class FakeTimesFMModel:
    def forecast(self, *, horizon, inputs):
        point = np.zeros((len(inputs), horizon), dtype=np.float32)
        raw = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
        raw[..., 0] = point
        for index in range(9):
            raw[..., index + 1] = index
        return point, raw


class ModelAdapterTest(unittest.TestCase):
    def test_normalized_output_shape_and_levels(self):
        point, raw = FakeTimesFMModel().forecast(horizon=4, inputs=[[1, 2], [3, 4]])
        result = normalize_forecast_output(point, raw)
        self.assertEqual(result["point_forecast"].shape, (2, 4))
        self.assertEqual(result["quantile_forecasts"].shape, (2, 4, 9))
        np.testing.assert_allclose(result["quantile_levels"], np.arange(1, 10) / 10)

    def test_adapter_forecast_accepts_variable_history_lengths(self):
        adapter = TimesFMAdapter.__new__(TimesFMAdapter)
        adapter.config = TimesFMConfig(context_length=3, max_horizon=8)
        adapter._model = FakeTimesFMModel()
        output = adapter.forecast([[1, 2], [1, 2, 3, 4]], prediction_length=4)
        self.assertEqual(output["point_forecast"].shape, (2, 4))
        self.assertEqual(output["quantile_forecasts"].shape, (2, 4, 9))

