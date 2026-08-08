"""GIFT-Eval integration using its official dataset split/evaluation objects."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Iterator, Mapping

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
    weighted_quantile_loss,
)
from src.models.timesfm import TIMESFM_25_QUANTILE_LEVELS


@dataclass(frozen=True)
class GiftEvalWindow:
    dataset: str
    series_id: str
    window_id: int
    history: np.ndarray
    ground_truth: np.ndarray
    timestamps: tuple[str | None, ...]
    frequency: str | None


@dataclass(frozen=True)
class WindowForecast:
    window: GiftEvalWindow
    point_forecast: np.ndarray
    quantile_forecasts: np.ndarray
    quantile_levels: np.ndarray
    elapsed_seconds: float


def load_gift_dataset(
    name: str,
    *,
    term: str = "short",
    to_univariate: bool = True,
    storage_env_var: str = "GIFT_EVAL",
) -> Any:
    """Load a dataset through ``gift_eval.data.Dataset`` from the official package."""

    import os

    if not os.getenv(storage_env_var):
        raise RuntimeError(
            f"{storage_env_var} is not set. Download Salesforce/GiftEval and set "
            "`GIFT_EVAL=/path/to/GiftEval` before running the benchmark."
        )
    try:
        from gift_eval.data import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "GIFT-Eval is not installed. Install requirements-lock.txt to use the official loader."
        ) from exc
    # The official transformation is intended for genuinely multivariate
    # targets.  Applying MultivariateToUnivariate to an already-univariate
    # target iterates over scalar values and produces zero-dimensional targets,
    # which GluonTS cannot split into test windows.  Inspect the official
    # dataset metadata first, then request the transformation only when it is
    # needed.
    dataset = Dataset(name=name, term=term, to_univariate=False, storage_env_var=storage_env_var)
    if to_univariate and int(dataset.target_dim) > 1:
        dataset = Dataset(name=name, term=term, to_univariate=True, storage_env_var=storage_env_var)
    return dataset


def _to_1d_target(value: Any, field: str) -> np.ndarray:
    target = np.asarray(value, dtype=np.float32)
    if target.ndim != 1:
        raise ValueError(
            f"{field} is not univariate (shape={target.shape}). Use to_univariate=True "
            "so the official GIFT-Eval flattening protocol is preserved."
        )
    if target.size == 0:
        raise ValueError(f"{field} is empty")
    return target


def _string_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if hasattr(value, "to_timestamp"):
            value = value.to_timestamp()
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _future_timestamps(
    input_entry: Mapping[str, Any],
    label_entry: Mapping[str, Any],
    history_length: int,
    horizon: int,
    frequency: str | None,
) -> tuple[str | None, ...]:
    start = label_entry.get("start")
    if start is None and input_entry.get("start") is not None:
        try:
            start = input_entry["start"] + history_length
        except TypeError:
            start = pd.Timestamp(input_entry["start"]) + pd.tseries.frequencies.to_offset(frequency)
    if start is None:
        return tuple([None] * horizon)
    try:
        if hasattr(start, "to_timestamp"):
            values = [start + offset for offset in range(horizon)]
            return tuple(_string_timestamp(value) for value in values)
        values = pd.date_range(start=pd.Timestamp(start), periods=horizon, freq=frequency)
        return tuple(value.isoformat() for value in values)
    except (TypeError, ValueError):
        return tuple([_string_timestamp(start)] + [None] * (horizon - 1))


def iter_test_windows(
    dataset: Any,
    *,
    dataset_name: str,
    max_series: int | None = None,
    max_windows: int | None = None,
) -> Iterator[GiftEvalWindow]:
    """Iterate official rolling test windows without shuffling or future leakage."""

    test_data = dataset.test_data
    seen_series: set[str] = set()
    yielded = 0
    for input_entry, label_entry in zip(test_data.input, test_data.label):
        series_id = str(input_entry.get("item_id", label_entry.get("item_id", len(seen_series))))
        if series_id not in seen_series:
            if max_series is not None and len(seen_series) >= max_series:
                break
            seen_series.add(series_id)
        if max_windows is not None and yielded >= max_windows:
            break

        history = _to_1d_target(input_entry["target"], "input.target")
        future = _to_1d_target(label_entry["target"], "label.target")
        horizon = int(dataset.prediction_length)
        if future.size < horizon:
            raise AssertionError(
                f"Official label for {series_id} has {future.size} points, expected at least {horizon}"
            )
        # GluonTS/GIFT-Eval labels can contain more than the requested horizon;
        # only the generated future window is evaluated.
        future = future[-horizon:]
        frequency = input_entry.get("freq", label_entry.get("freq"))
        yield GiftEvalWindow(
            dataset=f"{dataset_name}/{getattr(getattr(dataset, 'term', 'short'), 'value', getattr(dataset, 'term', 'short'))}",
            series_id=series_id,
            window_id=yielded,
            history=history,
            ground_truth=future,
            timestamps=_future_timestamps(
                input_entry, label_entry, history.size, horizon, frequency
            ),
            frequency=str(frequency) if frequency is not None else None,
        )
        yielded += 1


def seasonality_from_frequency(frequency: str | None) -> int:
    if not frequency:
        return 1
    key = frequency.upper()
    if key in {"S", "SEC", "SECOND", "5S", "10S"}:
        return 60
    if key in {"T", "MIN", "5T", "10T", "15T", "30T"}:
        return 24 * 60 // max(1, int(key[:-1]) if key[:-1].isdigit() else 1)
    if key in {"H", "HOURLY"}:
        return 24
    if key in {"D", "DAILY"}:
        return 7
    if key in {"W", "WEEKLY"}:
        return 52
    if key in {"M", "MONTHLY"}:
        return 12
    if key in {"Q", "QUARTERLY"}:
        return 4
    return 1


def forecast_windows(
    adapter: Any,
    windows: Iterable[GiftEvalWindow],
    *,
    prediction_length: int,
    batch_size: int = 1,
) -> list[WindowForecast]:
    windows = list(windows)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    results: list[WindowForecast] = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        histories = [window.history for window in chunk]
        started = time.perf_counter()
        output = adapter.forecast(histories, prediction_length)
        elapsed = max(0.0, time.perf_counter() - started)
        point = np.asarray(output["point_forecast"])
        quantiles = np.asarray(output["quantile_forecasts"])
        levels = np.asarray(output["quantile_levels"], dtype=float)
        if point.shape != (len(chunk), prediction_length):
            raise AssertionError(f"point forecast shape {point.shape} does not match chunk")
        if quantiles.shape[:2] != (len(chunk), prediction_length):
            raise AssertionError(f"quantile forecast shape {quantiles.shape} does not match chunk")
        per_window_elapsed = elapsed / max(1, len(chunk))
        for index, window in enumerate(chunk):
            results.append(
                WindowForecast(
                    window=window,
                    point_forecast=point[index].astype(np.float32, copy=False),
                    quantile_forecasts=quantiles[index].astype(np.float32, copy=False),
                    quantile_levels=levels.copy(),
                    elapsed_seconds=per_window_elapsed,
                )
            )
    return results


def _level_index(levels: np.ndarray, target: float) -> int | None:
    indices = np.flatnonzero(np.isclose(levels, target, atol=1e-8))
    return int(indices[0]) if indices.size else None


def aggregate_diagnostics(results: list[WindowForecast]) -> dict[str, Any]:
    """Aggregate custom diagnostics while leaving official metric names untouched."""

    if not results:
        return {}
    targets = np.stack([item.window.ground_truth for item in results])
    points = np.stack([item.point_forecast for item in results])
    quantiles = np.stack([item.quantile_forecasts for item in results])
    levels = results[0].quantile_levels
    if any(not np.array_equal(levels, item.quantile_levels) for item in results[1:]):
        raise ValueError("quantile levels changed within a run")
    seasonality = seasonality_from_frequency(results[0].window.frequency)
    per_window_mase = [
        mase(
            item.window.ground_truth[None, :],
            item.point_forecast[None, :],
            item.window.history[None, :],
            seasonality=seasonality,
        )
        for item in results
    ]
    metrics: dict[str, Any] = {
        "num_series": len({item.window.series_id for item in results}),
        "num_windows": len(results),
        "horizon": int(targets.shape[1]),
        "mae": mae(targets, points),
        "rmse": rmse(targets, points),
        "mase": float(np.nanmean(per_window_mase)) if np.any(np.isfinite(per_window_mase)) else None,
        "mean_pinball_loss": pinball_loss(targets, quantiles, levels),
        "diagnostic_weighted_quantile_loss": weighted_quantile_loss(targets, quantiles, levels),
        "quantile_crossing_rate": quantile_crossing_rate(quantiles),
        "runtime_seconds": float(np.sum([item.elapsed_seconds for item in results])),
        "average_latency_per_forecast_window_seconds": float(
            np.mean([item.elapsed_seconds for item in results])
        ),
        "throughput_windows_per_second": float(
            len(results) / np.sum([item.elapsed_seconds for item in results])
        )
        if np.sum([item.elapsed_seconds for item in results]) > 0
        else None,
    }
    q10 = _level_index(levels, 0.1)
    q90 = _level_index(levels, 0.9)
    if q10 is not None and q90 is not None:
        lower, upper = quantiles[..., q10], quantiles[..., q90]
        metrics["coverage_80"] = interval_coverage(targets, lower, upper)
        metrics["width_80"] = interval_width(lower, upper)
    else:
        metrics["coverage_80"] = None
        metrics["width_80"] = None
    # TimesFM 2.5 exposes q10..q90, not q05/q95; do not fabricate a 90% interval.
    metrics["coverage_90"] = None
    metrics["width_90"] = None
    metrics["quantile_levels"] = levels.tolist()
    return metrics


def official_gift_eval_metrics(dataset: Any, adapter: Any, *, batch_size: int = 512) -> dict[str, Any]:
    """Run GIFT-Eval's recommended GluonTS evaluator on a fresh test iterator."""

    try:
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss  # type: ignore[import-not-found]
        from gluonts.model.evaluation import evaluate_model  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The official GIFT-Eval evaluator requires its pinned GluonTS dependency."
        ) from exc

    predictor = _TimesFMPredictor(adapter, prediction_length=int(dataset.prediction_length))
    seasonality = seasonality_from_frequency(getattr(dataset, "freq", None))
    # GluonTS 0.15.1 requires the levels explicitly.  These are the nine
    # quantiles returned by the official TimesFM 2.5 continuous head; the
    # point/mean channel is not passed as a quantile level.
    metrics = [
        MASE(),
        MeanWeightedSumQuantileLoss(quantile_levels=TIMESFM_25_QUANTILE_LEVELS.tolist()),
    ]
    values = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=batch_size,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=seasonality,
    )
    if hasattr(values, "to_dict"):
        values = values.to_dict()
    return {str(key): _json_scalar(value) for key, value in dict(values).items()}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        # evaluate_model returns a one-row DataFrame; DataFrame.to_dict()
        # therefore wraps each scalar as {None: value}.  Unwrap that stable
        # representation while retaining a normal mapping for any future
        # multi-index evaluator output.
        if len(value) == 1 and next(iter(value)) is None:
            return _json_scalar(next(iter(value.values())))
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


class _TimesFMPredictor:
    """Minimal GluonTS Predictor facade used only by the official evaluator."""

    def __init__(self, adapter: Any, prediction_length: int):
        self.adapter = adapter
        self.prediction_length = prediction_length

    def predict(self, dataset: Iterable[Mapping[str, Any]], **_: Any) -> Iterator[Any]:
        try:
            from gluonts.model.forecast import QuantileForecast  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("GluonTS is required for the official evaluator") from exc
        for item in dataset:
            history = _to_1d_target(item["target"], "input.target")
            output = self.adapter.forecast([history], self.prediction_length)
            quantiles = np.asarray(output["quantile_forecasts"])[0].T
            levels = np.asarray(output["quantile_levels"], dtype=float)
            yield QuantileForecast(
                forecast_arrays=quantiles,
                start_date=item["start"] + len(history),
                item_id=item.get("item_id"),
                forecast_keys=[str(level) for level in levels],
            )
