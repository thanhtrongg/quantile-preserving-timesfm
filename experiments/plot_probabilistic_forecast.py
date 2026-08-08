"""Plot one stored forecast for sanity checking and later paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.storage import infer_quantile_levels, quantile_column, read_forecasts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--series-id", default=None)
    parser.add_argument("--window-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install the project dependencies with `pip install -e .`."
        ) from exc
    forecasts = read_forecasts(args.run / "forecasts.parquet")
    contexts = pd.read_parquet(args.run / "contexts.parquet")
    series_id = args.series_id or str(forecasts.iloc[0]["series_id"])
    window_id = args.window_id if args.window_id is not None else int(forecasts.iloc[0]["forecast_window_id"])
    forecast = forecasts[
        (forecasts["series_id"].astype(str) == series_id)
        & (forecasts["forecast_window_id"] == window_id)
    ].sort_values("time_step")
    context = contexts[
        (contexts["series_id"].astype(str) == series_id)
        & (contexts["forecast_window_id"] == window_id)
    ]
    if forecast.empty or context.empty:
        raise ValueError(f"No stored forecast/context for series_id={series_id!r}, window_id={window_id}")
    history = np.asarray(context.iloc[0]["history_values"], dtype=float)
    levels = infer_quantile_levels(forecast)
    x_history = np.arange(history.size)
    x_future = history.size + np.arange(len(forecast))
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(x_history, history, color="#263238", linewidth=1.2, label="history/context")
    axis.plot(x_future, forecast["ground_truth"], color="#1565c0", linewidth=1.8, label="ground truth")
    axis.plot(x_future, forecast["point_forecast"], color="#c62828", linewidth=1.6, label="point forecast")
    q10 = quantile_column(0.1)
    q90 = quantile_column(0.9)
    if q10 in forecast.columns and q90 in forecast.columns:
        axis.fill_between(
            x_future,
            forecast[q10].to_numpy(float),
            forecast[q90].to_numpy(float),
            color="#ef9a9a",
            alpha=0.35,
            label="80% interval (q10–q90)",
        )
    axis.axvline(history.size - 0.5, color="#78909c", linestyle="--", linewidth=1)
    axis.set_title(f"{forecast.iloc[0]['dataset']} — {series_id}, window {window_id}")
    axis.set_xlabel("time step")
    axis.set_ylabel("value")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = args.output or args.run / "forecast_sanity.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
