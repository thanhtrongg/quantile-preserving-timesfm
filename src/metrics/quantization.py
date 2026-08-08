"""Diagnostics for comparing precision-specific forecast artifacts."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.metrics.diagnostics import (
    interval_coverage,
    interval_width,
    mae,
    mase,
    pinball_loss,
    quantile_crossing_rate,
    rmse,
)
from src.evaluation.gift_eval import seasonality_from_frequency
from src.utils.storage import infer_quantile_levels, quantile_column


ALIGNMENT_KEYS = ["dataset", "series_id", "forecast_window_id", "time_step"]


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ALIGNMENT_KEYS + ["ground_truth", "point_forecast"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast artifact is missing columns: {missing}")
    return frame.sort_values(ALIGNMENT_KEYS).reset_index(drop=True)


def assert_forecast_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """Require identical identifiers, labels, and row ordering across variants."""

    left = _sorted_frame(reference)
    right = _sorted_frame(candidate)
    if left[ALIGNMENT_KEYS].to_dict("records") != right[ALIGNMENT_KEYS].to_dict("records"):
        raise AssertionError("Forecast identifiers do not align across precisions")
    if not np.array_equal(left["ground_truth"].to_numpy(), right["ground_truth"].to_numpy()):
        raise AssertionError("Ground truth differs across precision artifacts")
    if set(infer_quantile_levels(left)) != set(infer_quantile_levels(right)):
        raise AssertionError("Quantile levels differ across precision artifacts")


def assert_context_alignment(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    keys = ["dataset", "series_id", "forecast_window_id"]
    left = reference.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    if left[keys].to_dict("records") != right[keys].to_dict("records"):
        raise AssertionError("Context identifiers do not align across precisions")
    for index, (left_values, right_values) in enumerate(
        zip(left["history_values"], right["history_values"])
    ):
        if not np.array_equal(np.asarray(left_values), np.asarray(right_values)):
            raise AssertionError(f"Context differs across precision artifacts at row {index}")


def _group_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    frame = _sorted_frame(frame)
    levels = infer_quantile_levels(frame)
    groups = []
    targets = []
    points = []
    quantiles = []
    for (series_id, window_id), group in frame.groupby(
        ["series_id", "forecast_window_id"], sort=True
    ):
        group = group.sort_values("time_step")
        groups.append((str(series_id), int(window_id)))
        targets.append(group["ground_truth"].to_numpy(dtype=np.float64))
        points.append(group["point_forecast"].to_numpy(dtype=np.float64))
        quantiles.append(
            group[[quantile_column(float(level)) for level in levels]].to_numpy(dtype=np.float64)
        )
    return np.stack(targets), np.stack(points), np.stack(quantiles), groups


def per_window_diagnostics(
    frame: pd.DataFrame,
    contexts: pd.DataFrame,
    *,
    frequency: str | None,
    dataset: str,
    precision: str,
) -> pd.DataFrame:
    targets, points, quantiles, groups = _group_arrays(frame)
    levels = infer_quantile_levels(frame)
    context_keys = ["series_id", "forecast_window_id"]
    context_map = {
        (str(row.series_id), int(row.forecast_window_id)): np.asarray(row.history_values, dtype=float)
        for row in contexts.itertuples(index=False)
    }
    seasonality = seasonality_from_frequency(frequency)
    q10 = int(np.flatnonzero(np.isclose(levels, 0.1))[0])
    q90 = int(np.flatnonzero(np.isclose(levels, 0.9))[0])
    rows = []
    for index, (series_id, window_id) in enumerate(groups):
        history = context_map[(series_id, window_id)]
        rows.append(
            {
                "dataset": dataset,
                "precision": precision,
                "series_id": series_id,
                "forecast_window_id": window_id,
                "mae": mae(targets[index : index + 1], points[index : index + 1]),
                "rmse": rmse(targets[index : index + 1], points[index : index + 1]),
                "mase": mase(
                    targets[index : index + 1],
                    points[index : index + 1],
                    history[None, :],
                    seasonality,
                ),
                "mean_pinball_loss": pinball_loss(
                    targets[index : index + 1], quantiles[index : index + 1], levels
                ),
                "coverage_80": interval_coverage(
                    targets[index], quantiles[index, :, q10], quantiles[index, :, q90]
                ),
                "width_80": interval_width(
                    quantiles[index, :, q10], quantiles[index, :, q90]
                ),
                "quantile_crossing_rate": quantile_crossing_rate(quantiles[index]),
            }
        )
    return pd.DataFrame(rows)


def quantile_distortion(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    dataset: str,
    precision: str,
) -> pd.DataFrame:
    """Return mean absolute low-bit-vs-FP32 deviation for every quantile."""

    assert_forecast_alignment(reference, candidate)
    left = _sorted_frame(reference)
    right = _sorted_frame(candidate)
    levels = infer_quantile_levels(left)
    rows = []
    for level in levels:
        column = quantile_column(float(level))
        absolute = np.abs(right[column].to_numpy(float) - left[column].to_numpy(float))
        rows.append(
            {
                "dataset": dataset,
                "precision": precision,
                "quantile": float(level),
                "mean_absolute_quantile_deviation": float(np.mean(absolute)),
                "median_absolute_quantile_deviation": float(np.median(absolute)),
            }
        )

    lower = [quantile_column(float(level)) for level in levels if level <= 0.2]
    upper = [quantile_column(float(level)) for level in levels if level >= 0.8]
    median = quantile_column(0.5)
    width_reference = left[quantile_column(0.9)] - left[quantile_column(0.1)]
    width_candidate = right[quantile_column(0.9)] - right[quantile_column(0.1)]
    rows.append(
        {
            "dataset": dataset,
            "precision": precision,
            "quantile": "summary",
            "mean_absolute_quantile_deviation": float(
                np.mean(np.abs(right[median] - left[median]))
            ),
            "median_absolute_quantile_deviation": float(
                np.mean(
                    np.abs(
                        right[lower].to_numpy(float) - left[lower].to_numpy(float)
                    )
                )
            ),
            "median_q50_deviation": float(np.mean(np.abs(right[median] - left[median]))),
            "lower_tail_q10_q20_deviation": float(
                np.mean(np.abs(right[lower].to_numpy(float) - left[lower].to_numpy(float)))
            ),
            "upper_tail_q80_q90_deviation": float(
                np.mean(np.abs(right[upper].to_numpy(float) - left[upper].to_numpy(float)))
            ),
            "mean_q10_q90_interval_width_change": float(
                np.mean(width_candidate - width_reference)
            ),
            "relative_q10_q90_interval_width_change": float(
                np.mean((width_candidate - width_reference) / np.maximum(np.abs(width_reference), 1e-12))
            ),
        }
    )
    return pd.DataFrame(rows)
