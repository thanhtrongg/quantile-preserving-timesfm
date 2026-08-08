import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.storage import (
    context_rows,
    forecast_rows,
    infer_quantile_levels,
    read_forecasts,
    write_parquet,
)


class StorageTest(unittest.TestCase):
    def test_forecast_artifact_round_trip_keeps_dynamic_levels(self):
        levels = np.asarray([0.1, 0.5, 0.9])
        frame = forecast_rows(
            dataset="toy/1H/short",
            series_ids=["s0"],
            window_ids=[0],
            timestamps=[["2026-01-01T00:00:00", "2026-01-01T01:00:00"]],
            targets=np.asarray([[1.0, 2.0]]),
            point_forecasts=np.asarray([[1.1, 1.9]]),
            quantile_forecasts=np.asarray([[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]]),
            quantile_levels=levels,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecasts.parquet"
            write_parquet(frame, path)
            loaded = read_forecasts(path)
        np.testing.assert_allclose(infer_quantile_levels(loaded), levels)
        self.assertEqual(len(loaded), 2)

    def test_context_artifact_is_arrow_serializable(self):
        frame = context_rows(
            dataset="toy",
            series_ids=["s0"],
            window_ids=[0],
            histories=[np.asarray([1.0, 2.0])],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.parquet"
            write_parquet(frame, path)
            self.assertTrue(path.exists())

