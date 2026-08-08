"""Portable Parquet artifacts for forecasts and contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def quantile_column(level: float) -> str:
    return f"q_{float(level):.10g}"


def forecast_rows(
    *,
    dataset: str,
    series_ids: Iterable[str],
    window_ids: Iterable[int],
    timestamps: Iterable[Iterable[str | None]],
    targets: np.ndarray,
    point_forecasts: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray,
) -> pd.DataFrame:
    series_ids = list(series_ids)
    window_ids = list(window_ids)
    timestamp_rows = list(timestamps)
    y = np.asarray(targets)
    point = np.asarray(point_forecasts)
    quantiles = np.asarray(quantile_forecasts)
    levels = np.asarray(quantile_levels, dtype=float)
    if y.ndim != 2 or point.shape != y.shape or quantiles.shape[:2] != y.shape:
        raise ValueError("target, point, and quantile arrays have incompatible shapes")
    if quantiles.shape[-1] != levels.size:
        raise ValueError("quantile levels do not match forecast tensor")
    rows: list[dict[str, Any]] = []
    for b, (series_id, window_id) in enumerate(zip(series_ids, window_ids)):
        row_timestamps = timestamp_rows[b] if b < len(timestamp_rows) else [None] * y.shape[1]
        if len(row_timestamps) != y.shape[1]:
            raise ValueError("timestamp row does not match forecast horizon")
        for step in range(y.shape[1]):
            row: dict[str, Any] = {
                "dataset": dataset,
                "series_id": str(series_id),
                "forecast_window_id": int(window_id),
                "time_step": int(step),
                "timestamp": row_timestamps[step],
                "ground_truth": float(y[b, step]),
                "point_forecast": float(point[b, step]),
            }
            for q_index, level in enumerate(levels):
                row[quantile_column(float(level))] = float(quantiles[b, step, q_index])
            rows.append(row)
    return pd.DataFrame(rows)


def context_rows(
    *,
    dataset: str,
    series_ids: Iterable[str],
    window_ids: Iterable[int],
    histories: Iterable[np.ndarray],
) -> pd.DataFrame:
    rows = []
    for series_id, window_id, history in zip(series_ids, window_ids, histories):
        values = np.asarray(history, dtype=float).reshape(-1)
        rows.append(
            {
                "dataset": dataset,
                "series_id": str(series_id),
                "forecast_window_id": int(window_id),
                "history_values": values.tolist(),
            }
        )
    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_forecasts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def infer_quantile_levels(frame: pd.DataFrame) -> np.ndarray:
    levels = []
    for column in frame.columns:
        if column.startswith("q_"):
            levels.append(float(column[2:]))
    if not levels:
        return np.asarray([], dtype=float)
    return np.asarray(sorted(levels), dtype=float)
