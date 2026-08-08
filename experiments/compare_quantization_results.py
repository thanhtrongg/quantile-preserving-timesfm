"""Join quantization artifacts and generate paper-ready comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.metrics.quantization import (
    assert_context_alignment,
    assert_forecast_alignment,
    quantile_distortion,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dataset_label(dataset_config: dict[str, Any]) -> str:
    return str(
        dataset_config.get(
            "label",
            f"{dataset_config['name']}/{dataset_config.get('term', 'short')}",
        )
    )


def _latest_run(root: Path, variant: str, dataset: str) -> Path | None:
    dataset_root = root / variant / "_".join(
        "".join(character.lower() if character.isalnum() else "_" for character in dataset).split("_")
    )
    candidates = sorted(
        [path for path in dataset_root.glob("run_*") if (path / "metrics.json").exists()],
        key=lambda path: path.name,
    )
    return candidates[-1] if candidates else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(metrics: dict[str, Any], key: str) -> Any:
    if key == "wql":
        return metrics.get("official_evaluator", {}).get("metrics", {}).get(
            "mean_weighted_sum_quantile_loss"
        )
    if key == "official_mase":
        return metrics.get("official_evaluator", {}).get("metrics", {}).get("MASE[0.5]")
    return metrics.get(key)


def _summary_row(dataset: str, variant: str, run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "precision": variant,
        "status": metrics.get("status"),
        "run_dir": str(run_dir),
        "num_series": metrics.get("num_series"),
        "num_windows": metrics.get("num_windows"),
        "horizon": metrics.get("horizon"),
        "official_mase": _metric(metrics, "official_mase"),
        "mase": metrics.get("mase"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "wql": _metric(metrics, "wql"),
        "mean_pinball_loss": metrics.get("mean_pinball_loss"),
        "coverage_80": metrics.get("coverage_80"),
        "coverage_80_delta_to_nominal": (
            metrics.get("coverage_80") - 0.8
            if metrics.get("coverage_80") is not None
            else None
        ),
        "coverage_80_abs_calibration_deviation": (
            abs(metrics.get("coverage_80") - 0.8)
            if metrics.get("coverage_80") is not None
            else None
        ),
        "width_80": metrics.get("width_80"),
        "quantile_crossing_rate": metrics.get("quantile_crossing_rate"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "runtime_seconds_including_load_and_eval": metrics.get(
            "runtime_seconds_including_load_and_eval"
        ),
        "average_latency_per_forecast_window_seconds": metrics.get(
            "average_latency_per_forecast_window_seconds"
        ),
        "throughput_windows_per_second": metrics.get("throughput_windows_per_second"),
        "peak_memory_mb": metrics.get("peak_memory_mb"),
        "cpu_ram_mb_at_artifact_write": metrics.get("cpu_ram_mb_at_artifact_write"),
        "parameter_storage_mb": metrics.get("parameter_storage_mb"),
        "serialized_model_size_mb": metrics.get("serialized_model_size_mb"),
        "official_evaluator_status": metrics.get("official_evaluator", {}).get("status"),
        "evaluation_scope": metrics.get("evaluation_scope", "full_official_test_windows"),
        "quantization_backend": metrics.get("quantization", {}).get("backend"),
    }


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "dataset",
        "precision",
        "status",
        "evaluation_scope",
        "official_mase",
        "wql",
        "mean_pinball_loss",
        "coverage_80",
        "coverage_80_delta_to_nominal",
        "coverage_80_abs_calibration_deviation",
        "width_80",
        "runtime_seconds",
        "peak_memory_mb",
        "parameter_storage_mb",
    ]
    view = frame[columns].copy()
    view.columns = [
        "Dataset",
        "Precision",
        "Status",
        "Scope",
        "MASE",
        "WQL",
        "Pinball",
        "Coverage80",
        "Coverage80 - 0.8",
        "Absolute calibration deviation",
        "Width80",
        "Runtime (s)",
        "Peak memory (MiB)",
        "Parameter storage (MiB)",
    ]
    path.write_text(view.to_markdown(index=False, floatfmt=".6f") + "\n", encoding="utf-8")


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "dataset",
        "precision",
        "official_mase",
        "wql",
        "mean_pinball_loss",
        "coverage_80",
        "coverage_80_delta_to_nominal",
        "coverage_80_abs_calibration_deviation",
        "width_80",
        "runtime_seconds",
        "peak_memory_mb",
    ]
    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Precision & MASE & WQL & Pinball & Coverage80 & Coverage80-0.8 & |Coverage80-0.8| & Width80 & Runtime (s) & Peak memory (MiB) \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        values = [_latex_escape(row[column]) if column in {"dataset", "precision"} else ("--" if pd.isna(row[column]) else f"{float(row[column]):.6f}") for column in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_comparisons(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    root = Path(config["runtime"].get("output_root", "results/milestone2"))
    comparison_root = root / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    variants = [str(item["name"]) for item in config["variants"]]
    datasets = [_dataset_label(item) for item in config["benchmark"]["datasets"]]

    rows = []
    run_map: dict[tuple[str, str], Path] = {}
    for dataset in datasets:
        for variant in variants:
            run_dir = _latest_run(root, variant, dataset.split("/")[0] if dataset.count("/") == 1 else dataset.rsplit("/", 1)[0])
            if run_dir is None:
                rows.append(
                    {
                        "dataset": dataset,
                        "precision": variant,
                        "status": "missing",
                        "evaluation_scope": "not_executed_full",
                    }
                )
                continue
            metrics = _read_json(run_dir / "metrics.json")
            run_map[(dataset, variant)] = run_dir
            rows.append(_summary_row(dataset, variant, run_dir, metrics))

    summary = pd.DataFrame(rows)
    summary.to_csv(comparison_root / "quantization_summary.csv", index=False)
    _write_markdown(summary, comparison_root / "quantization_summary.md")
    _write_latex(summary, comparison_root / "quantization_summary.tex")

    distortion_frames = []
    deltas = []
    alignment = []
    for dataset in datasets:
        reference_dir = run_map.get((dataset, "fp32"))
        if reference_dir is None or not (reference_dir / "forecasts.parquet").exists():
            alignment.append({"dataset": dataset, "precision": "fp32", "status": "missing_reference"})
            continue
        reference_forecasts = pd.read_parquet(reference_dir / "forecasts.parquet")
        reference_contexts = pd.read_parquet(reference_dir / "contexts.parquet")
        reference_row = summary[(summary.dataset == dataset) & (summary.precision == "fp32")].iloc[0]
        for variant in variants:
            candidate_dir = run_map.get((dataset, variant))
            if variant == "fp32" or candidate_dir is None:
                continue
            metrics = _read_json(candidate_dir / "metrics.json")
            if metrics.get("status") != "completed":
                alignment.append({"dataset": dataset, "precision": variant, "status": metrics.get("status")})
                continue
            candidate_forecasts = pd.read_parquet(candidate_dir / "forecasts.parquet")
            candidate_contexts = pd.read_parquet(candidate_dir / "contexts.parquet")
            assert_forecast_alignment(reference_forecasts, candidate_forecasts)
            assert_context_alignment(reference_contexts, candidate_contexts)
            alignment.append(
                {
                    "dataset": dataset,
                    "precision": variant,
                    "status": "passed",
                    "forecast_identifier_alignment": True,
                    "ground_truth_alignment": True,
                    "context_alignment": True,
                }
            )
            distortion_frames.append(
                quantile_distortion(
                    reference_forecasts,
                    candidate_forecasts,
                    dataset=dataset,
                    precision=variant,
                )
            )
            candidate_row = summary[(summary.dataset == dataset) & (summary.precision == variant)].iloc[0]
            deltas.append(
                {
                    "dataset": dataset,
                    "precision": variant,
                    "delta_mase": candidate_row.get("official_mase") - reference_row.get("official_mase"),
                    "delta_wql": candidate_row.get("wql") - reference_row.get("wql"),
                    "delta_coverage_80": candidate_row.get("coverage_80") - reference_row.get("coverage_80"),
                    "delta_coverage_80_pp": 100.0
                    * (candidate_row.get("coverage_80") - reference_row.get("coverage_80")),
                    "delta_width_80": candidate_row.get("width_80") - reference_row.get("width_80"),
                    "relative_mase_change": (candidate_row.get("official_mase") - reference_row.get("official_mase")) / reference_row.get("official_mase"),
                    "relative_wql_change": (candidate_row.get("wql") - reference_row.get("wql")) / reference_row.get("wql"),
                    "relative_width_change": (candidate_row.get("width_80") - reference_row.get("width_80")) / reference_row.get("width_80"),
                }
            )

    distortion = pd.concat(distortion_frames, ignore_index=True) if distortion_frames else pd.DataFrame()
    distortion.to_csv(comparison_root / "quantile_distortion.csv", index=False)
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(comparison_root / "quantization_deltas.csv", index=False)
    if not delta_frame.empty:
        delta_frame.to_markdown(
            comparison_root / "quantization_deltas.md", index=False, floatfmt=".6f"
        )
        delta_frame.to_latex(
            comparison_root / "quantization_deltas.tex", index=False, float_format="%.6f"
        )
    if not distortion.empty:
        distortion.to_markdown(
            comparison_root / "quantile_distortion.md", index=False, floatfmt=".6f"
        )
        distortion.to_latex(
            comparison_root / "quantile_distortion.tex", index=False, float_format="%.6f"
        )
    pd.DataFrame(alignment).to_json(comparison_root / "alignment_validation.json", orient="records", indent=2)
    return {
        "summary_path": str(comparison_root / "quantization_summary.csv"),
        "distortion_path": str(comparison_root / "quantile_distortion.csv"),
        "deltas_path": str(comparison_root / "quantization_deltas.csv"),
        "alignment_path": str(comparison_root / "alignment_validation.json"),
        "rows": len(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_comparisons(args.config), indent=2))


if __name__ == "__main__":
    main()
