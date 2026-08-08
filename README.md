# Quantile-Preserving Post-Training Quantization for Probabilistic Time-Series Foundation Models

This repository contains the research code and experimental pipeline for our work on Quantile-Preserving Post-Training Quantization for Probabilistic Time-Series Foundation Models. The current development stage is a reproducible low-bit quantization pilot for TimesFM 2.5 on GIFT-Eval.

**TimesFM 2.5 FP32 zero-shot probabilistic baseline on GIFT-Eval**

The baseline stores point forecasts, every quantile forecast returned by the
model, the corresponding quantile levels, ground truth, timestamps, context,
metrics, and environment metadata. It is the reference artifact for later
BF16, INT8, INT4, and proposed quantile-preserving PTQ comparisons.

The pilot evaluates FP32, BF16, and backend-supported INT8 weight-only
inference. INT4 is wired to the real TorchAO backend and is recorded as
unsupported on the CPU-only validation environment when its required kernel
dependency is unavailable. The proposed quantile-preserving PTQ method,
quantile-preserving loss, fine-tuning, QAT, mixed-precision search, and
dataset/window cherry-picking are not implemented.

## Pinned upstream implementations

- TimesFM PyTorch package: `2.0.2`, official repository commit
  `8a22ca28a0239d34c095b1eba7fea92d22198e0c`.
- TimesFM checkpoint: `google/timesfm-2.5-200m-pytorch`.
- GIFT-Eval package source: official repository commit
  `d8184bb51079bb5021332f8e5d7486c378a52202`; package metadata currently
  reports `0.0.0a0`.
- Official GluonTS evaluator dependency: `0.15.1`.
- TorchAO quantization backend: `0.18.0`.
- Table generation dependency: `tabulate`.

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

Run the Milestone 2 quantization pilot. This uses the same official rolling
windows, contexts, labels, horizon, and quantile levels for every variant:

```powershell
$env:GIFT_EVAL = (Resolve-Path .\data\GiftEval).Path
python -m experiments.run_quantization_pilot `
  --config configs/quantization_pilot.yaml
python -m experiments.compare_quantization_results `
  --config configs/quantization_pilot.yaml
python -m experiments.plot_quantization_impact `
  --comparison-root results/milestone2/comparisons
```

The pilot configuration evaluates `saugeenday/D/short` (20 windows) and
`bizitobs_l2c/5T/short` (7 effective univariate series × 20 windows = 140
windows), with context 1024, horizons 30 and 48 respectively, and q10--q90
at levels `[0.1, 0.2, ..., 0.9]`. The full Bitbrains configuration is retained
as a candidate, but its current `to_univariate` expansion produces 45,000
windows and is not used as primary CPU evidence.

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
  --run results/fp32/timesfm_2_5_200m/saugeenday_d/run_005
```

Verify a saved run in a fresh process without invoking TimesFM again:

```bash
$env:GIFT_EVAL = (Resolve-Path .\data\GiftEval).Path
python -m experiments.verify_saved_artifact `
  --run results/fp32/timesfm_2_5_200m/saugeenday_d/run_005 `
  --check-gift
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

## Current Status

### Current Stage: Low-Bit Quantization Pilot

Milestone 2 is complete for the supported pilot variants. On 2026-08-08, the
repository ran the official TimesFM 2.5 PyTorch checkpoint through the official
GIFT-Eval windows for `saugeenday/D/short` and `bizitobs_l2c/5T/short`:

| Variant | Backend/scheme | Saugeenday MASE / WQL | Bizitobs MASE / WQL | Parameter storage |
|---|---|---:|---:|---:|
| FP32 | Eager FP32 | 0.957658 / 0.399888 | 0.283652 / 0.078124 | 882.30 MiB |
| BF16 | BF16 weights + CPU BF16 autocast | 0.958503 / 0.399844 | 0.283834 / 0.078147 | 441.15 MiB |
| INT8 | TorchAO symmetric per-output-channel weight-only INT8, FP32 activations | 0.957529 / 0.400070 | 0.283545 / 0.078100 | 221.77 MiB |
| INT4 | TorchAO INT4 weight-only, group size 128 | unsupported | unsupported | not applicable |

All completed variants used the same context length (1024), horizons (30 and
48), official rolling windows (20 and 140), ground truth, and nine returned
quantiles q10--q90. Point shapes were `(20, 30)` and `(140, 48)`; quantile
shapes were `(20, 30, 9)` and `(140, 48, 9)`. Quantiles were finite and
non-decreasing in every saved window, and the crossing rate was 0.0 for all
completed runs. The INT4 result is an explicit capability limitation, not a
simulated or cast-to-INT4 result: TorchAO reported `Requires mslk >= 1.0.0`
for this CPU-only environment.

Relative to FP32, BF16 and INT8 changed official MASE and WQL by at most
0.09% and 0.05% respectively across these two pilot configurations. Their
quantile outputs nevertheless showed measurable small deviations: median
absolute q-level deviations ranged from 0.0423 to 0.1092 for INT8 and from
0.0717 to 0.0966 for BF16 in the two configurations; mean q10--q90 width
changes were positive in all completed comparisons. This is a WEAK SIGNAL for
the pilot research question: accuracy was broadly preserved, but two
configurations are insufficient for a general claim and INT4 remains
unvalidated. No Milestone 3 method, QAT, fine-tuning, or proposed PTQ method
has been started.

Fresh-process artifact verification reloaded each completed forecast without
running TimesFM again, reproduced the saved diagnostics, checked matching
series/window IDs, and matched the official GIFT-Eval contexts and labels.
Tables are generated under `results/milestone2/comparisons/`; figures are
generated as vector PDF and 300-dpi PNG under the same directory. The exact
backend rationale and limitations are documented in
`docs/quantization_backend.md`. Downloaded data and generated results remain
ignored by Git; Milestone 1 FP32 artifacts are preserved unchanged.

The completed run directories are under `results/milestone2/{fp32,bf16,int8}/`
for the two datasets; the corresponding `int4/` directories contain the
unsupported status and exact backend error.

Contamination status for every pilot/candidate dataset is documented in
`docs/gift_eval_selection.md`. Electricity remains development-only because
overlap with TimesFM pretraining or benchmark sources cannot be ruled out.
