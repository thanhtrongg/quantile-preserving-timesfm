"""Fresh-process verification for a saved forecast artifact.

This command never imports or invokes TimesFM.  It reloads the portable
Parquet/JSON files, recomputes the local diagnostics, and optionally compares
the stored contexts and labels with the official GIFT-Eval test windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.gift_eval import (
    iter_test_windows,
    load_gift_dataset,
    seasonality_from_frequency,
)
from src.metrics.diagnostics import (
    interval_coverage,
    interval_width,
    mae,
    mase,
    pinball_loss,
    quantile_crossing_rate,
    rmse,
    weighted_quantile_loss,
)
from src.utils.storage import infer_quantile_levels, quantile_column, read_forecasts


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify(run_dir: Path, check_gift: bool) -> dict[str, object]:
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        saved_metrics = json.load(handle)
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    forecasts = read_forecasts(run_dir / "forecasts.parquet")
    contexts = pd.read_parquet(run_dir / "contexts.parquet")
    levels = infer_quantile_levels(forecasts)
    if levels.size == 0:
        raise AssertionError("No quantile columns found in forecasts.parquet")

    resolved = config.get("resolved", config.get("benchmark", {}))
    horizon = int(resolved["prediction_length"] if "prediction_length" in resolved else resolved["resolved_prediction_length"])
    keys = (
        forecasts[["series_id", "forecast_window_id"]]
        .drop_duplicates()
        .sort_values(["series_id", "forecast_window_id"])
        .itertuples(index=False, name=None)
    )
    targets: list[np.ndarray] = []
    points: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    context_map = {
        (str(row.series_id), int(row.forecast_window_id)): np.asarray(row.history_values, dtype=np.float64)
        for row in contexts.itertuples(index=False)
    }
    expected_steps = np.arange(horizon)
    for series_id, window_id in keys:
        key = (str(series_id), int(window_id))
        frame = forecasts[
            (forecasts["series_id"] == series_id)
            & (forecasts["forecast_window_id"] == window_id)
        ].sort_values("time_step")
        if not np.array_equal(frame["time_step"].to_numpy(), expected_steps):
            raise AssertionError(f"Non-contiguous time_step values for {key}")
        if key not in context_map:
            raise AssertionError(f"Missing context for {key}")
        targets.append(frame["ground_truth"].to_numpy(dtype=np.float64))
        points.append(frame["point_forecast"].to_numpy(dtype=np.float64))
        quantiles.append(
            frame[[quantile_column(float(level)) for level in levels]].to_numpy(dtype=np.float64)
        )
        histories.append(context_map[key])

    y = np.stack(targets)
    point = np.stack(points)
    q = np.stack(quantiles)
    history = histories
    if not (np.isfinite(y).all() and np.isfinite(point).all() and np.isfinite(q).all()):
        raise AssertionError("NaN or infinite values found in saved forecasts")
    crossing = quantile_crossing_rate(q)
    frequency = resolved.get("frequency", config.get("benchmark", {}).get("resolved_frequency"))
    seasonality = seasonality_from_frequency(frequency)
    per_window_mase = [
        mase(y[index : index + 1], point[index : index + 1], history[index][None, :], seasonality)
        for index in range(y.shape[0])
    ]
    q10 = int(np.flatnonzero(np.isclose(levels, 0.1))[0]) if np.any(np.isclose(levels, 0.1)) else None
    q90 = int(np.flatnonzero(np.isclose(levels, 0.9))[0]) if np.any(np.isclose(levels, 0.9)) else None
    if q10 is None or q90 is None:
        coverage_80 = None
        width_80 = None
    else:
        coverage_80 = interval_coverage(y, q[..., q10], q[..., q90])
        width_80 = interval_width(q[..., q10], q[..., q90])
    recomputed = {
        "mae": mae(y, point),
        "rmse": rmse(y, point),
        "mase": float(np.nanmean(per_window_mase)),
        "mean_pinball_loss": pinball_loss(y, q, levels),
        "diagnostic_weighted_quantile_loss": weighted_quantile_loss(y, q, levels),
        "coverage_80": coverage_80,
        "width_80": width_80,
        "quantile_crossing_rate": crossing,
    }
    comparisons = {}
    for key, value in recomputed.items():
        saved = saved_metrics.get(key)
        comparisons[key] = {
            "saved": _json_value(saved),
            "recomputed": _json_value(value),
            "match": bool(saved is not None and np.isclose(float(saved), float(value), rtol=1e-6, atol=1e-6)),
        }
    if not all(item["match"] for item in comparisons.values()):
        raise AssertionError(f"Saved diagnostic metrics do not reproduce: {comparisons}")

    official_check = {"status": "not_requested"}
    if check_gift:
        dataset_cfg = config.get("dataset", config.get("benchmark", {}).get("dataset"))
        if dataset_cfg is None:
            raise AssertionError("Artifact config does not identify its dataset")
        dataset = load_gift_dataset(
            dataset_cfg["name"],
            term=dataset_cfg["term"],
            to_univariate=bool(config.get("benchmark", {}).get("to_univariate", True)),
            storage_env_var=str(config.get("benchmark", {}).get("storage_env_var", "GIFT_EVAL")),
        )
        official_windows = list(iter_test_windows(dataset, dataset_name=dataset_cfg["name"]))
        if len(official_windows) != len(targets):
            raise AssertionError("Saved window count differs from official GIFT-Eval windows")
        for window in official_windows:
            key = (window.series_id, window.window_id)
            frame = forecasts[
                (forecasts["series_id"] == window.series_id)
                & (forecasts["forecast_window_id"] == window.window_id)
            ].sort_values("time_step")
            if not np.array_equal(context_map[key], window.history):
                raise AssertionError(f"Context mismatch/future leakage check failed for {key}")
            if not np.array_equal(frame["ground_truth"].to_numpy(dtype=np.float32), window.ground_truth):
                raise AssertionError(f"Ground-truth mismatch for {key}")
        official_check = {
            "status": "passed",
            "windows_compared": len(official_windows),
            "context_matches_pre_forecast_input": True,
            "ground_truth_matches_official_labels": True,
        }

    return {
        "run_dir": str(run_dir),
        "point_tensor_shape": list(point.shape),
        "quantile_tensor_shape": list(q.shape),
        "context_lengths": sorted({int(values.size) for values in history}),
        "quantile_levels": levels.tolist(),
        "finite_forecasts": True,
        "quantiles_non_decreasing": bool(np.all(q[..., :-1] <= q[..., 1:])),
        "diagnostic_comparisons": comparisons,
        "official_data_check": official_check,
        "sample_forecasts": [
            {
                "series_id": str(forecasts.iloc[0]["series_id"]),
                "window_id": int(forecasts.iloc[0]["forecast_window_id"]),
                "target": forecasts.iloc[:3]["ground_truth"].tolist(),
                "point": forecasts.iloc[:3]["point_forecast"].tolist(),
                "q10": forecasts.iloc[:3][quantile_column(0.1)].tolist(),
                "q50": forecasts.iloc[:3][quantile_column(0.5)].tolist(),
                "q90": forecasts.iloc[:3][quantile_column(0.9)].tolist(),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--check-gift", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.check_gift), indent=2, default=_json_value))


if __name__ == "__main__":
    main()
