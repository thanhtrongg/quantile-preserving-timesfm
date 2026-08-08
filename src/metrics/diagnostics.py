"""Small NumPy metrics used alongside the official GIFT-Eval metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(np.asarray(arrays[0], dtype=np.float64), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _validate_quantiles(quantile_forecasts: np.ndarray, quantile_levels: Iterable[float]) -> np.ndarray:
    forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    levels = np.asarray(list(quantile_levels), dtype=np.float64)
    if forecasts.ndim < 1 or forecasts.shape[-1] != levels.size:
        raise ValueError(
            f"quantile forecast last dimension {forecasts.shape[-1:]!r} does not match "
            f"levels {levels.shape}"
        )
    if levels.size == 0 or not np.all(np.isfinite(levels)) or not np.all(np.diff(levels) > 0):
        raise ValueError("quantile_levels must be finite and strictly increasing")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("quantile_levels must lie strictly between 0 and 1")
    return levels


def pinball_loss(
    targets: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: Iterable[float],
) -> float:
    """Mean pinball loss over all finite target/forecast/level entries."""

    y = np.asarray(targets, dtype=np.float64)[..., None]
    q = np.asarray(quantile_forecasts, dtype=np.float64)
    levels = _validate_quantiles(q, quantile_levels)
    if y.shape[:-1] != q.shape[:-1]:
        raise ValueError(f"targets shape {y.shape[:-1]} does not match forecasts {q.shape[:-1]}")
    errors = y - q
    losses = np.maximum(levels * errors, (levels - 1.0) * errors)
    mask = np.isfinite(y) & np.isfinite(q)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(losses[mask]))


def weighted_quantile_loss(
    targets: np.ndarray,
    quantile_forecasts: np.ndarray,
    quantile_levels: Iterable[float],
) -> float:
    """A transparent normalized weighted quantile loss diagnostic.

    GIFT-Eval's authoritative probabilistic result is retained from its own
    evaluator.  This function is only a local diagnostic for row-level output
    and intentionally does not replace the official metric.
    """

    y = np.asarray(targets, dtype=np.float64)
    q = np.asarray(quantile_forecasts, dtype=np.float64)
    levels = _validate_quantiles(q, quantile_levels)
    if y.shape != q.shape[:-1]:
        raise ValueError(f"targets shape {y.shape} does not match forecasts {q.shape[:-1]}")
    errors = y[..., None] - q
    losses = 2.0 * np.maximum(levels * errors, (levels - 1.0) * errors)
    mask = np.isfinite(y) & np.all(np.isfinite(q), axis=-1)
    denominator = np.sum(np.abs(y[mask]))
    if denominator == 0 or not np.any(mask):
        return float("nan")
    return float(np.sum(losses[mask]) / (q.shape[-1] * denominator))


def interval_coverage(targets: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a closed prediction interval."""

    y = np.asarray(targets, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError(f"interval arrays must share a shape, got {y.shape}, {lo.shape}, {hi.shape}")
    if np.any(lo > hi):
        raise ValueError("interval lower bound exceeds upper bound")
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of a prediction interval."""

    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"interval arrays must share a shape, got {lo.shape} and {hi.shape}")
    if np.any(lo > hi):
        raise ValueError("interval lower bound exceeds upper bound")
    values = hi - lo
    mask = np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def quantile_crossing_rate(quantile_forecasts: np.ndarray) -> float:
    """Fraction of adjacent quantile pairs that cross."""

    q = np.asarray(quantile_forecasts, dtype=np.float64)
    if q.ndim < 2 or q.shape[-1] < 2:
        raise ValueError("quantile_forecasts must have at least two quantile levels")
    crossing = q[..., :-1] > q[..., 1:]
    mask = np.isfinite(q[..., :-1]) & np.isfinite(q[..., 1:])
    if not np.any(mask):
        return float("nan")
    return float(np.mean(crossing[mask]))


def mae(targets: np.ndarray, forecasts: np.ndarray) -> float:
    y = np.asarray(targets, dtype=np.float64)
    f = np.asarray(forecasts, dtype=np.float64)
    if y.shape != f.shape:
        raise ValueError(f"targets and forecasts must share a shape, got {y.shape} and {f.shape}")
    mask = np.isfinite(y) & np.isfinite(f)
    return float(np.mean(np.abs(y[mask] - f[mask]))) if np.any(mask) else float("nan")


def rmse(targets: np.ndarray, forecasts: np.ndarray) -> float:
    y = np.asarray(targets, dtype=np.float64)
    f = np.asarray(forecasts, dtype=np.float64)
    if y.shape != f.shape:
        raise ValueError(f"targets and forecasts must share a shape, got {y.shape} and {f.shape}")
    mask = np.isfinite(y) & np.isfinite(f)
    return float(np.sqrt(np.mean((y[mask] - f[mask]) ** 2))) if np.any(mask) else float("nan")


def mase(
    targets: np.ndarray,
    forecasts: np.ndarray,
    history: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean absolute scaled error using the pre-forecast history only."""

    y = np.asarray(targets, dtype=np.float64)
    f = np.asarray(forecasts, dtype=np.float64)
    h = np.asarray(history, dtype=np.float64)
    if y.shape != f.shape or y.ndim != 2 or h.ndim != 2 or h.shape[0] != y.shape[0]:
        raise ValueError("targets/forecasts must be (B,H) and history must be (B,T)")
    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    if h.shape[1] <= seasonality:
        return float("nan")
    scale_values = np.abs(h[:, seasonality:] - h[:, :-seasonality])
    scales = np.nanmean(scale_values, axis=1)
    errors = np.nanmean(np.abs(y - f), axis=1)
    valid = np.isfinite(scales) & np.isfinite(errors) & (scales > 0)
    return float(np.mean(errors[valid] / scales[valid])) if np.any(valid) else float("nan")
