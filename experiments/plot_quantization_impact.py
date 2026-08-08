"""Generate publication-ready Milestone 2 comparison figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.storage import quantile_column


PRECISION_ORDER = ["fp32", "bf16", "int8", "int4"]
COLORS = {"fp32": "#263238", "bf16": "#1565c0", "int8": "#ef6c00", "int4": "#6a1b9a"}


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["precision"] = pd.Categorical(frame["precision"], PRECISION_ORDER, ordered=True)
    return frame.sort_values(["dataset", "precision"])


def _save(figure: plt.Figure, output_root: Path, stem: str) -> None:
    figure.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_root / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_panel(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str, ylabel: str) -> None:
    for dataset, group in frame.groupby("dataset", sort=False):
        values = []
        for precision in PRECISION_ORDER:
            row = group[group["precision"] == precision]
            values.append(row[column].iloc[0] if not row.empty else np.nan)
        axis.plot(
            PRECISION_ORDER,
            values,
            marker="o",
            linewidth=1.8,
            label=str(dataset),
        )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("Precision")


def plot_impact(comparison_root: Path) -> None:
    summary = _ordered(pd.read_csv(comparison_root / "quantization_summary.csv"))
    completed = summary[summary["status"] == "completed"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_panel(axes[0, 0], completed, "official_mase", "Point accuracy", "Official MASE")
    _plot_panel(axes[0, 1], completed, "wql", "Probabilistic accuracy", "Mean weighted quantile loss")
    _plot_panel(axes[1, 0], completed, "coverage_80", "Interval calibration", "q10–q90 coverage")
    _plot_panel(axes[1, 1], completed, "parameter_storage_mb", "Parameter storage", "MiB")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=max(1, len(labels)))
    _save(figure, comparison_root, "quantization_impact")


def plot_distortion(comparison_root: Path) -> None:
    distortion = pd.read_csv(comparison_root / "quantile_distortion.csv")
    distortion = distortion[distortion["quantile"] != "summary"].copy()
    distortion["quantile"] = distortion["quantile"].astype(float)
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for (dataset, precision), group in distortion.groupby(["dataset", "precision"], sort=False):
        group = group.sort_values("quantile")
        axis.plot(
            group["quantile"],
            group["mean_absolute_quantile_deviation"],
            marker="o",
            linewidth=1.8,
            color=COLORS.get(str(precision), None),
            label=f"{dataset} — {precision.upper()}",
        )
    axis.set_xlabel("Quantile level")
    axis.set_ylabel("Mean absolute deviation from FP32")
    axis.set_xticks(sorted(distortion["quantile"].unique()))
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    _save(figure, comparison_root, "quantile_distortion")


def _latest_run(root: Path, precision: str, dataset: str) -> Path:
    dataset_key = dataset.rsplit("/", 1)[0] if dataset.rsplit("/", 1)[-1] in {"short", "medium", "long"} else dataset
    slug = "_".join("".join(c.lower() if c.isalnum() else "_" for c in dataset_key).split("_"))
    runs = sorted((root / precision / slug).glob("run_*/forecasts.parquet"))
    if not runs:
        raise FileNotFoundError(f"No forecast artifact for {precision}/{dataset}")
    return runs[-1].parent


def plot_example(comparison_root: Path, dataset: str, window_id: int, output_root: Path) -> None:
    root = comparison_root.parent
    fp32_dir = _latest_run(root, "fp32", dataset)
    lowbit = None
    for precision in ["int4", "int8", "bf16"]:
        try:
            candidate = _latest_run(root, precision, dataset)
            with (candidate / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get("status") == "completed":
                lowbit = (precision, candidate)
                break
        except (FileNotFoundError, ValueError):
            continue
    if lowbit is None:
        raise RuntimeError("No completed low-bit artifact is available for example figure")
    precision, lowbit_dir = lowbit
    fp = pd.read_parquet(fp32_dir / "forecasts.parquet")
    low = pd.read_parquet(lowbit_dir / "forecasts.parquet")
    fp = fp[(fp["forecast_window_id"] == window_id)].sort_values("time_step")
    low = low[(low["forecast_window_id"] == window_id)].sort_values("time_step")
    context = pd.read_parquet(fp32_dir / "contexts.parquet")
    context = context[context["forecast_window_id"] == window_id].iloc[0]
    history = np.asarray(context["history_values"], dtype=float)[-1024:]
    x_history = np.arange(history.size)
    x_future = history.size + np.arange(len(fp))
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.plot(x_history, history, color="#455a64", linewidth=1.1, label="Historical context")
    axis.plot(x_future, fp["ground_truth"], color="#1565c0", linewidth=1.7, label="Ground truth")
    axis.plot(x_future, fp[quantile_column(0.5)], color="#263238", linewidth=1.7, label="FP32 median")
    axis.plot(x_future, low[quantile_column(0.5)], color="#ef6c00", linewidth=1.7, label=f"{precision.upper()} median")
    axis.fill_between(x_future, fp[quantile_column(0.1)], fp[quantile_column(0.9)], color="#90caf9", alpha=0.3, label="FP32 q10–q90")
    axis.fill_between(x_future, low[quantile_column(0.1)], low[quantile_column(0.9)], color="#ffcc80", alpha=0.3, label=f"{precision.upper()} q10–q90")
    axis.axvline(history.size - 0.5, color="#78909c", linestyle="--", linewidth=1)
    axis.set_title(f"{dataset} — window {window_id}")
    axis.set_xlabel("Forecast step")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", ncol=2)
    _save(figure, output_root, "example_forecast_lowbit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=Path("results/milestone2/comparisons"))
    parser.add_argument("--dataset", default="saugeenday/D/short")
    parser.add_argument("--window-id", type=int, default=0)
    args = parser.parse_args()
    plot_impact(args.comparison_root)
    plot_distortion(args.comparison_root)
    plot_example(args.comparison_root, args.dataset, args.window_id, args.comparison_root)
    print(args.comparison_root)


if __name__ == "__main__":
    main()
