"""Run the reproducible FP32 TimesFM 2.5 pilot on official GIFT-Eval data."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.gift_eval import (
    aggregate_diagnostics,
    forecast_windows,
    iter_test_windows,
    load_gift_dataset,
    official_gift_eval_metrics,
)
from src.models.timesfm import TimesFMAdapter, TimesFMConfig
from src.utils.reproducibility import environment_metadata, seed_everything, write_json
from src.utils.storage import context_rows, forecast_rows, write_parquet


PINNED_TIMESFM_COMMIT = "8a22ca28a0239d34c095b1eba7fea92d22198e0c"
PINNED_GIFT_EVAL_COMMIT = "d8184bb51079bb5021332f8e5d7486c378a52202"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def _slug(value: str) -> str:
    return "_".join("".join(character.lower() if character.isalnum() else "_" for character in value).split("_"))


def _next_run_dir(root: Path, dataset_name: str, model_name: str) -> Path:
    dataset_root = root / "fp32" / _slug(model_name) / _slug(dataset_name)
    dataset_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        int(path.name.removeprefix("run_"))
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("run_") and path.name.removeprefix("run_").isdigit()
    ]
    run_dir = dataset_root / f"run_{(max(candidates, default=0) + 1):03d}"
    run_dir.mkdir()
    return run_dir


def _model_parameter_size_mb(adapter: TimesFMAdapter) -> float | None:
    model = adapter.model
    try:
        total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        return float(total_bytes / (1024**2))
    except (AttributeError, TypeError):
        return None


def _peak_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except ImportError:
        pass
    return None


def _cpu_ram_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except ImportError:
        return None


def _run_dataset(
    *,
    adapter: TimesFMAdapter,
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    output_root: Path,
    max_series: int | None,
    max_windows: int | None,
    batch_size: int,
    skip_official_evaluator: bool,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    runtime = config["runtime"]
    dataset_name = str(dataset_config["name"])
    term = str(dataset_config.get("term", "short"))
    run_dir = _next_run_dir(output_root, dataset_name, str(config["model"]["name"]))

    started = time.perf_counter()
    dataset = load_gift_dataset(
        dataset_name,
        term=term,
        to_univariate=bool(benchmark.get("to_univariate", True)),
        storage_env_var=str(benchmark.get("storage_env_var", "GIFT_EVAL")),
    )
    windows = list(
        iter_test_windows(
            dataset,
            dataset_name=dataset_name,
            max_series=max_series,
            max_windows=max_windows,
        )
    )
    if not windows:
        raise RuntimeError(f"The official GIFT-Eval loader returned no windows for {dataset_name}/{term}")
    prediction_length = int(benchmark.get("prediction_length") or dataset.prediction_length)
    if prediction_length != int(dataset.prediction_length):
        raise ValueError(
            "Config prediction_length must be null or exactly the official GIFT-Eval horizon; "
            f"got {prediction_length}, official={dataset.prediction_length}"
        )

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    forecasts = forecast_windows(
        adapter,
        windows,
        prediction_length=prediction_length,
        batch_size=batch_size,
    )
    metrics = aggregate_diagnostics(forecasts)
    metrics["dataset"] = f"{dataset_name}/{term}"
    metrics["official_evaluator"] = {}
    if skip_official_evaluator:
        metrics["official_evaluator"] = {"status": "skipped_by_cli"}
    elif max_series is not None or max_windows is not None:
        metrics["official_evaluator"] = {
            "status": "skipped_for_limited_run",
            "reason": "official evaluator always consumes the complete official test iterator",
        }
    else:
        try:
            metrics["official_evaluator"] = {
                "status": "completed",
                "metrics": official_gift_eval_metrics(dataset, adapter, batch_size=512),
            }
        except Exception as exc:  # preserve artifacts so the incompatibility is inspectable
            metrics["official_evaluator"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    levels = forecasts[0].quantile_levels
    y = [item.window.ground_truth for item in forecasts]
    point = [item.point_forecast for item in forecasts]
    quantiles = [item.quantile_forecasts for item in forecasts]
    rows = forecast_rows(
        dataset=f"{dataset_name}/{term}",
        series_ids=[item.window.series_id for item in forecasts],
        window_ids=[item.window.window_id for item in forecasts],
        timestamps=[item.window.timestamps for item in forecasts],
        targets=pd.DataFrame(y).to_numpy(),
        point_forecasts=pd.DataFrame(point).to_numpy(),
        quantile_forecasts=np.stack(quantiles),
        quantile_levels=levels,
    )
    write_parquet(rows, run_dir / "forecasts.parquet")
    contexts = context_rows(
        dataset=f"{dataset_name}/{term}",
        series_ids=[item.window.series_id for item in forecasts],
        window_ids=[item.window.window_id for item in forecasts],
        histories=[item.window.history for item in forecasts],
    )
    write_parquet(contexts, run_dir / "contexts.parquet")

    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "runtime_seconds_including_load_and_eval": elapsed,
            "peak_memory_mb": _peak_memory_mb(),
            "cpu_ram_mb_at_artifact_write": _cpu_ram_mb(),
            "checkpoint_model_size_mb": _model_parameter_size_mb(adapter),
        }
    )
    run_config = {
        **config,
        "benchmark": {**benchmark, "dataset": dataset_config, "resolved_prediction_length": prediction_length},
        "reproducibility": {
            "timesfm_version": "2.0.2",
            "timesfm_repository_commit": PINNED_TIMESFM_COMMIT,
            "gift_eval_version": "0.0.0a0",
            "gift_eval_repository_commit": PINNED_GIFT_EVAL_COMMIT,
        },
    }
    write_json(run_dir / "config.json", run_config)
    write_json(
        run_dir / "environment.json",
        environment_metadata(
            timesfm_metadata=adapter.metadata(),
            gift_eval_version="0.0.0a0",
            gift_eval_commit=PINNED_GIFT_EVAL_COMMIT,
        ),
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "summary.json",
        {
            "dataset": f"{dataset_name}/{term}",
            "run_dir": str(run_dir),
            "num_series": metrics["num_series"],
            "num_windows": metrics["num_windows"],
            "horizon": metrics["horizon"],
            "quantile_levels": levels.tolist(),
            "artifacts": ["config.json", "environment.json", "metrics.json", "forecasts.parquet", "contexts.parquet"],
        },
    )
    return {**metrics, "run_dir": str(run_dir)}


def _summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "dataset",
        "num_series",
        "num_windows",
        "horizon",
        "mase",
        "mae",
        "rmse",
        "wql",
        "mean_pinball_loss",
        "coverage_80",
        "width_80",
        "coverage_90",
        "width_90",
        "quantile_crossing_rate",
        "runtime_seconds",
        "peak_memory_mb",
        "average_latency_per_forecast_window_seconds",
        "throughput_windows_per_second",
        "run_dir",
    ]
    row = {key: metrics.get(key) for key in keys}
    row["wql"] = metrics.get("official_evaluator", {}).get("metrics", {}).get(
        "mean_weighted_sum_quantile_loss"
    )
    official = metrics.get("official_evaluator", {}).get("metrics", {})
    if isinstance(official, dict):
        row.update(official)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--skip-official-evaluator", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_yaml(args.config)
    model_config = config["model"]
    runtime = config["runtime"]
    seed_everything(int(runtime.get("seed", 0)))
    adapter = TimesFMAdapter(
        TimesFMConfig(
            checkpoint=str(model_config["checkpoint"]),
            dtype=str(model_config.get("dtype", "float32")),
            context_length=int(model_config.get("context_length", 1024)),
            max_horizon=int(model_config.get("max_horizon", 256)),
            quantile_head=bool(model_config.get("quantile_head", True)),
            fix_quantile_crossing=bool(model_config.get("fix_quantile_crossing", True)),
            normalize_inputs=bool(model_config.get("normalize_inputs", True)),
            force_flip_invariance=bool(model_config.get("force_flip_invariance", True)),
            infer_is_positive=bool(model_config.get("infer_is_positive", False)),
            device=str(runtime.get("device", "auto")),
        )
    )
    output_root = Path(runtime.get("output_root", "results"))
    max_series = args.max_series if args.max_series is not None else runtime.get("max_series")
    max_windows = args.max_windows if args.max_windows is not None else runtime.get("max_windows")
    batch_size = int(runtime.get("batch_size", 1))
    all_rows = []
    for dataset_config in config["benchmark"]["datasets"]:
        result = _run_dataset(
            adapter=adapter,
            config=config,
            dataset_config=dataset_config,
            output_root=output_root,
            max_series=int(max_series) if max_series is not None else None,
            max_windows=int(max_windows) if max_windows is not None else None,
            batch_size=batch_size,
            skip_official_evaluator=args.skip_official_evaluator,
        )
        all_rows.append(_summary_row(result))
        print(json.dumps({"dataset": result["dataset"], "run_dir": result["run_dir"]}, sort_keys=True))

    summary_path = output_root / "fp32_baseline_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(all_rows)
    if summary_path.exists():
        old_frame = pd.read_csv(summary_path)
        new_frame = pd.concat([old_frame, new_frame], ignore_index=True)
    new_frame.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
