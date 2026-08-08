"""Run the Milestone 2 low-bit TimesFM observation pilot."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
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
)
from src.metrics.quantization import per_window_diagnostics
from src.models.timesfm import TimesFMAdapter, TimesFMConfig
from src.utils.reproducibility import environment_metadata, seed_everything, write_json
from src.utils.storage import context_rows, forecast_rows, write_parquet


TIMESFM_COMMIT = "8a22ca28a0239d34c095b1eba7fea92d22198e0c"
GIFT_EVAL_COMMIT = "d8184bb51079bb5021332f8e5d7486c378a52202"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def _slug(value: str) -> str:
    return "_".join(
        "".join(character.lower() if character.isalnum() else "_" for character in value).split("_")
    )


def _dataset_label(dataset_config: dict[str, Any]) -> str:
    return str(
        dataset_config.get(
            "label",
            f"{dataset_config['name']}/{dataset_config.get('term', 'short')}",
        )
    )


def _dataset_run_key(dataset_label: str) -> str:
    dataset_key = (
        dataset_label.rsplit("/", 1)[0]
        if dataset_label.rsplit("/", 1)[-1] in {"short", "medium", "long"}
        else dataset_label
    )
    return _slug(dataset_key)


def _next_run_dir(root: Path, variant: str, dataset_name: str) -> Path:
    dataset_root = root / variant / _dataset_run_key(dataset_name)
    dataset_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        int(path.name.removeprefix("run_"))
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("run_") and path.name.removeprefix("run_").isdigit()
    ]
    run_dir = dataset_root / f"run_{(max(candidates, default=0) + 1):03d}"
    run_dir.mkdir()
    return run_dir


def _tensor_bytes(value: Any) -> int:
    if hasattr(value, "qdata"):
        total = _tensor_bytes(value.qdata)
        for name in ("scale", "zero_point", "act_quant_scale", "act_quant_zero_point", "act_pre_scale"):
            item = getattr(value, name, None)
            if item is not None:
                total += _tensor_bytes(item)
        return total
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    return 0


def _parameter_storage_mb(adapter: TimesFMAdapter) -> float | None:
    module = getattr(adapter.model, "model", adapter.model)
    try:
        total = sum(_tensor_bytes(value) for value in module.state_dict().values())
        return float(total / (1024**2))
    except (AttributeError, TypeError, RuntimeError):
        return None


def _serialized_state_size_mb(adapter: TimesFMAdapter, directory: Path) -> float | None:
    """Measure serialized state size without retaining a second model artifact."""

    # TorchAO tensor subclasses intentionally expose quantized storage through
    # qdata/scales rather than a plain state_dict tensor.  torch.save on these
    # subclasses can materialize the dequantized representation and is not a
    # meaningful serialized-size measurement for this pilot.  The exact
    # quantized parameter storage is logged separately.
    if adapter.config.quantization_backend != "none":
        return None
    try:
        import torch

        state = getattr(adapter.model, "model", adapter.model).state_dict()
        with tempfile.NamedTemporaryFile(
            prefix="quant_state_", suffix=".pt", dir=directory, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            torch.save(state, temporary_path)
            return float(temporary_path.stat().st_size / (1024**2))
        finally:
            temporary_path.unlink(missing_ok=True)
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError):
        return None


def _peak_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except ImportError:
        pass
    try:
        import psutil

        memory = psutil.Process().memory_info()
        return float(getattr(memory, "peak_wset", memory.rss) / (1024**2))
    except (ImportError, AttributeError):
        return None


def _cpu_ram_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except ImportError:
        return None


def _make_adapter(model_config: dict[str, Any], variant: dict[str, Any], runtime: dict[str, Any]) -> TimesFMAdapter:
    return TimesFMAdapter(
        TimesFMConfig(
            checkpoint=str(model_config["checkpoint"]),
            dtype=str(variant.get("dtype", "float32")),
            context_length=int(model_config.get("context_length", 1024)),
            max_horizon=int(model_config.get("max_horizon", 256)),
            quantile_head=bool(model_config.get("quantile_head", True)),
            fix_quantile_crossing=bool(model_config.get("fix_quantile_crossing", True)),
            normalize_inputs=bool(model_config.get("normalize_inputs", True)),
            force_flip_invariance=bool(model_config.get("force_flip_invariance", True)),
            infer_is_positive=bool(model_config.get("infer_is_positive", False)),
            device=str(runtime.get("device", "auto")),
            torch_compile=bool(variant.get("torch_compile", True)),
            quantization_backend=str(variant.get("quantization_backend", "none")),
            quantization_group_size=int(variant.get("quantization_group_size", 128)),
        )
    )


def _write_unsupported(
    *,
    run_dir: Path,
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    variant: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    metrics = {
        "status": "unsupported",
        "precision": variant["name"],
        "dataset": _dataset_label(dataset_config),
        "error": f"{type(error).__name__}: {error}",
        "quantization": {
            "backend": variant.get("quantization_backend", "none"),
            "group_size": variant.get("quantization_group_size"),
        },
    }
    write_json(run_dir / "config.json", {**config, "variant": variant, "dataset": dataset_config})
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "summary.json", {"status": "unsupported", "run_dir": str(run_dir)})
    return {**metrics, "run_dir": str(run_dir)}


def _run_dataset(
    *,
    adapter: TimesFMAdapter,
    config: dict[str, Any],
    variant: dict[str, Any],
    dataset_config: dict[str, Any],
    output_root: Path,
    model_load_seconds: float,
    parameter_storage_mb: float | None,
    serialized_state_size_mb: float | None,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    runtime = config["runtime"]
    variant_name = str(variant["name"])
    dataset_name = str(dataset_config["name"])
    term = str(dataset_config.get("term", "short"))
    dataset_label = _dataset_label(dataset_config)
    run_dir = _next_run_dir(output_root, variant_name, dataset_label)
    started = time.perf_counter()
    dataset = load_gift_dataset(
        dataset_name,
        term=term,
        to_univariate=bool(benchmark.get("to_univariate", True)),
        storage_env_var=str(benchmark.get("storage_env_var", "GIFT_EVAL")),
    )
    max_series = benchmark.get("max_series")
    max_windows = benchmark.get("max_windows")
    windows = list(
        iter_test_windows(
            dataset,
            dataset_name=dataset_name,
            max_series=int(max_series) if max_series is not None else None,
            max_windows=int(max_windows) if max_windows is not None else None,
        )
    )
    if not windows:
        raise RuntimeError(f"The official GIFT-Eval loader returned no windows for {dataset_name}/{term}")
    prediction_length = int(benchmark.get("prediction_length") or dataset.prediction_length)
    if prediction_length != int(dataset.prediction_length):
        raise ValueError("Configured horizon does not match official GIFT-Eval horizon")

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
        batch_size=int(runtime.get("batch_size", 1)),
    )
    metrics = aggregate_diagnostics(forecasts)
    metrics["status"] = "completed"
    metrics["precision"] = variant_name
    metrics["dataset"] = dataset_label
    metrics["model_load_seconds"] = model_load_seconds
    metrics["evaluation_scope"] = (
        "bounded_probe"
        if max_series is not None or max_windows is not None
        else "full_official_test_windows"
    )
    metrics["parameter_storage_mb"] = parameter_storage_mb
    metrics["serialized_model_size_mb"] = serialized_state_size_mb
    metrics["serialized_model_size_note"] = (
        "not measured for TorchAO tensor subclasses; see parameter_storage_mb"
        if serialized_state_size_mb is None and adapter.config.quantization_backend != "none"
        else "temporary torch.save state size"
    )
    metrics["quantization"] = dict(adapter.quantization_metadata())
    if max_series is not None or max_windows is not None:
        metrics["official_evaluator"] = {
            "status": "not_run",
            "reason": "bounded probe; official full-dataset aggregate is not defined for a subset",
        }
    else:
        try:
            metrics["official_evaluator"] = {
                "status": "completed",
                "metrics": official_gift_eval_metrics(
                    dataset,
                    adapter,
                    batch_size=512,
                    precomputed_forecasts=forecasts,
                ),
            }
        except Exception as exc:
            metrics["official_evaluator"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    levels = forecasts[0].quantile_levels
    targets = np.stack([item.window.ground_truth for item in forecasts])
    points = np.stack([item.point_forecast for item in forecasts])
    quantiles = np.stack([item.quantile_forecasts for item in forecasts])
    rows = forecast_rows(
        dataset=dataset_label,
        series_ids=[item.window.series_id for item in forecasts],
        window_ids=[item.window.window_id for item in forecasts],
        timestamps=[item.window.timestamps for item in forecasts],
        targets=targets,
        point_forecasts=points,
        quantile_forecasts=quantiles,
        quantile_levels=levels,
    )
    write_parquet(rows, run_dir / "forecasts.parquet")
    contexts = context_rows(
        dataset=dataset_label,
        series_ids=[item.window.series_id for item in forecasts],
        window_ids=[item.window.window_id for item in forecasts],
        histories=[item.window.history for item in forecasts],
    )
    write_parquet(contexts, run_dir / "contexts.parquet")
    per_window_diagnostics(
        rows,
        contexts,
        frequency=str(dataset.freq),
        dataset=f"{dataset_name}/{term}",
        precision=variant_name,
    ).to_csv(run_dir / "per_window_metrics.csv", index=False)

    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "runtime_seconds_including_load_and_eval": elapsed,
            "peak_memory_mb": _peak_memory_mb(),
            "cpu_ram_mb_at_artifact_write": _cpu_ram_mb(),
        }
    )
    run_config = {
        **config,
        "variant": variant,
        "dataset": dataset_config,
        "resolved": {
            "prediction_length": prediction_length,
            "frequency": str(dataset.freq),
            "official_windows_per_series": int(dataset.windows),
            "num_series": metrics["num_series"],
            "num_windows": metrics["num_windows"],
            "evaluation_scope": (
                "bounded_probe"
                if max_series is not None or max_windows is not None
                else "full_official_test_windows"
            ),
        },
        "reproducibility": {
            "timesfm_version": "2.0.2",
            "timesfm_repository_commit": TIMESFM_COMMIT,
            "gift_eval_version": "0.0.0a0",
            "gift_eval_repository_commit": GIFT_EVAL_COMMIT,
        },
    }
    write_json(run_dir / "config.json", run_config)
    environment = environment_metadata(
        timesfm_metadata=adapter.metadata(),
        gift_eval_version="0.0.0a0",
        gift_eval_commit=GIFT_EVAL_COMMIT,
    )
    environment["quantization"] = adapter.quantization_metadata()
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "precision": variant_name,
            "dataset": dataset_label,
            "run_dir": str(run_dir),
            "num_series": metrics["num_series"],
            "num_windows": metrics["num_windows"],
            "evaluation_scope": (
                "bounded_probe"
                if max_series is not None or max_windows is not None
                else "full_official_test_windows"
            ),
            "horizon": metrics["horizon"],
            "quantile_levels": levels.tolist(),
            "artifacts": [
                "config.json",
                "environment.json",
                "metrics.json",
                "forecasts.parquet",
                "contexts.parquet",
                "per_window_metrics.csv",
            ],
        },
    )
    return {**metrics, "run_dir": str(run_dir)}


def _run_variant(config_path: Path, config: dict[str, Any], variant: dict[str, Any]) -> None:
    runtime = config["runtime"]
    seed_everything(int(runtime.get("seed", 0)))
    output_root = Path(runtime.get("output_root", "results/milestone2"))
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        started = time.perf_counter()
        adapter = _make_adapter(config["model"], variant, runtime)
        model_load_seconds = time.perf_counter() - started
        parameter_storage_mb = _parameter_storage_mb(adapter)
        serialized_state_size_mb = _serialized_state_size_mb(adapter, output_root)
    except Exception as exc:
        print(json.dumps({"variant": variant["name"], "status": "unsupported", "error": str(exc)}))
        for dataset_config in config["benchmark"]["datasets"]:
            run_dir = _next_run_dir(
                output_root,
                str(variant["name"]),
                _dataset_label(dataset_config),
            )
            _write_unsupported(
                run_dir=run_dir,
                config=config,
                dataset_config=dataset_config,
                variant=variant,
                error=exc,
            )
        return

    for dataset_config in config["benchmark"]["datasets"]:
        result = _run_dataset(
            adapter=adapter,
            config=config,
            variant=variant,
            dataset_config=dataset_config,
            output_root=output_root,
            model_load_seconds=model_load_seconds,
            parameter_storage_mb=parameter_storage_mb,
            serialized_state_size_mb=serialized_state_size_mb,
        )
        print(json.dumps({"dataset": result["dataset"], "precision": result["precision"], "run_dir": result["run_dir"]}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Run only the matching dataset name or public label; may be repeated.",
    )
    parser.add_argument("--max-series", type=int, default=None, help="Bound a real probe by series count.")
    parser.add_argument("--max-windows", type=int, default=None, help="Bound a real probe by window count.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override the configured artifact root.")
    parser.add_argument("--no-subprocess", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_yaml(args.config)
    if args.dataset:
        requested = set(args.dataset)
        datasets = [
            dataset
            for dataset in config["benchmark"]["datasets"]
            if str(dataset["name"]) in requested or _dataset_label(dataset) in requested
        ]
        if len(datasets) != len(requested):
            available = sorted(
                {
                    str(dataset["name"])
                    for dataset in config["benchmark"]["datasets"]
                }
                | {_dataset_label(dataset) for dataset in config["benchmark"]["datasets"]}
            )
            raise ValueError(f"Unknown dataset selector(s): {sorted(requested)}; available={available}")
        config = {
            **config,
            "benchmark": {**config["benchmark"], "datasets": datasets},
        }
    if args.max_series is not None or args.max_windows is not None:
        config = {
            **config,
            "benchmark": {
                **config["benchmark"],
                "max_series": args.max_series,
                "max_windows": args.max_windows,
            },
        }
    if args.output_root is not None:
        config = {**config, "runtime": {**config["runtime"], "output_root": str(args.output_root)}}
    variants = list(config.get("variants", []))
    if args.variant is not None:
        variants = [variant for variant in variants if variant["name"] == args.variant]
        if not variants:
            raise ValueError(f"Unknown variant: {args.variant}")
        _run_variant(args.config, config, variants[0])
        return

    for variant in variants:
        if args.no_subprocess or args.max_series is not None or args.max_windows is not None or args.output_root is not None:
            _run_variant(args.config, config, variant)
            continue
        subprocess.run(
            [sys.executable, "-m", "experiments.run_quantization_pilot", "--config", str(args.config), "--variant", str(variant["name"])],
            check=True,
            env=os.environ.copy(),
        )


if __name__ == "__main__":
    main()
