# Quantile-Preserving Post-Training Quantization for Probabilistic Time-Series Foundation Models

This repository contains the research code and experimental pipeline for our work on Quantile-Preserving Post-Training Quantization for Probabilistic Time-Series Foundation Models. The current development stage focuses on establishing a reproducible FP32 TimesFM 2.5 baseline on GIFT-Eval.

**TimesFM 2.5 FP32 zero-shot probabilistic baseline on GIFT-Eval**

The baseline stores point forecasts, every quantile forecast returned by the
model, the corresponding quantile levels, ground truth, timestamps, context,
metrics, and environment metadata. It is the reference artifact for later
FP16, INT8, INT4, and proposed quantile-preserving PTQ comparisons.

Not implemented in this milestone: FP16 comparison, INT8, INT4, custom PTQ,
quantile-preserving loss, fine-tuning, QAT, mixed-precision search, or any
dataset/window cherry-picking.

## Pinned upstream implementations

- TimesFM PyTorch package: `2.0.2`, official repository commit
  `8a22ca28a0239d34c095b1eba7fea92d22198e0c`.
- TimesFM checkpoint: `google/timesfm-2.5-200m-pytorch`.
- GIFT-Eval package source: official repository commit
  `d8184bb51079bb5021332f8e5d7486c378a52202`; package metadata currently
  reports `0.0.0a0`.

The adapter uses the official TimesFM 2.5 API:
`TimesFM_2p5_200M_torch.from_pretrained`, `ForecastConfig`, `compile`, and
`forecast`. With the continuous quantile head enabled, the official output is
`(B, H, 10)`: point/mean plus q10 through q90. The normalized artifact stores
the point separately and stores quantiles as `(B, H, 9)` with levels
`[0.1, ..., 0.9]`.

## Reproduce from a clean environment

Python 3.10+ is required. Select a CPU or CUDA PyTorch wheel appropriate for
the machine, then run:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[runtime,dev]"
```

Download the official GIFT-Eval dataset and point the loader at it:

```bash
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ./data/GiftEval
# PowerShell
$env:GIFT_EVAL = (Resolve-Path .\data\GiftEval).Path
# Linux/macOS: export GIFT_EVAL="$PWD/data/GiftEval"
```

Run the unit/smoke checks:

```bash
python -m unittest discover -s tests -v
```

Run the five-dataset FP32 pilot:

```bash
python -m experiments.run_fp32_gift_eval `
  --config configs/fp32_timesfm_gift.yaml
```

For a bounded pipeline check against one real GIFT-Eval dataset, use limits
and explicitly skip the full official aggregate evaluator:

```bash
python -m experiments.run_fp32_gift_eval `
  --config configs/fp32_timesfm_gift.yaml `
  --max-series 1 `
  --max-windows 1 `
  --skip-official-evaluator
```

That bounded command is a real GIFT-Eval loader/model run when dependencies,
checkpoint, and data are present; it is not the full benchmark result. The
official evaluator is run automatically for an uncapped pilot. If it fails,
the run records the exact error under `metrics.json` rather than hiding it.

Inspect metrics and plot one stored forecast:

```bash
Get-Content results/fp32_baseline_summary.csv
python -m experiments.plot_probabilistic_forecast `
  --run results/fp32/timesfm_2_5_200m/electricity_15t/run_001
```

## Artifact layout

Each dataset run is isolated:

```text
results/
├── fp32_baseline_summary.csv
└── fp32/timesfm_2_5_200m/<dataset>/run_001/
    ├── config.json
    ├── environment.json
    ├── metrics.json
    ├── forecasts.parquet
    ├── contexts.parquet
    ├── summary.json
    └── forecast_sanity.png       # after plotting
```

`forecasts.parquet` has one row per series/window/time step and dynamic
`q_<level>` columns. `contexts.parquet` is a portable Arrow list column used
only to reproduce the context portion of sanity plots; no Python objects are
serialized.

Primary benchmark metrics are retained under the native GIFT-Eval names from
its GluonTS evaluator. Local diagnostics are mean pinball loss, weighted
quantile loss diagnostic, q10–q90 coverage and width, and quantile crossing
rate. MASE, MAE, and RMSE are also recorded for row-level analysis. No custom
composite score is used.

## Current reproducibility status

The repository contains the real TimesFM/GIFT-Eval integration and tests, but
Milestone 1 is only complete after a real uncapped GIFT-Eval run succeeds.
The source workspace used to create this scaffold has no downloaded checkpoint
or GIFT-Eval dataset, so no benchmark numbers are committed or claimed here.
