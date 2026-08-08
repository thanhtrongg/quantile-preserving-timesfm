# Quantile-Preserving Post-Training Quantization for Probabilistic Time-Series Foundation Models

This repository contains the reproducible research pipeline for characterizing
low-bit TimesFM 2.5 probabilistic forecasts on GIFT-Eval. The proposed
quantile-preserving PTQ method has not been implemented.

## Upstream implementations

- TimesFM PyTorch package `2.0.2`, commit `8a22ca28a0239d34c095b1eba7fea92d22198e0c`
- Checkpoint `google/timesfm-2.5-200m-pytorch`
- GIFT-Eval commit `d8184bb51079bb5021332f8e5d7486c378a52202`
- GluonTS `0.15.1`
- TorchAO `0.18.0`

The official TimesFM 2.5 quantile head returns point/mean plus q10 through
q90. Artifacts store the point forecast as `(B, H)` and the nine quantile
forecasts as `(B, H, 9)` with levels `[0.1, 0.2, ..., 0.9]`.

## Environment and setup

Python 3.10+ is required. Install a PyTorch wheel suitable for the machine,
then install the repository runtime dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[runtime,dev]"
```

Download the official GIFT-Eval data and set its loader path:

```bash
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ./data/GiftEval
# PowerShell
$env:GIFT_EVAL = (Resolve-Path .\data\GiftEval).Path
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Reproduction commands

Milestone 2's supported pilot remains reproducible with:

```powershell
$env:GIFT_EVAL = (Resolve-Path .\data\GiftEval).Path
python -m experiments.run_quantization_pilot --config configs/quantization_pilot.yaml
python -m experiments.compare_quantization_results --config configs/quantization_pilot.yaml
python -m experiments.plot_quantization_impact --comparison-root results/milestone2/comparisons
```

Milestone 3 uses the four selected GIFT-Eval configurations:

```powershell
$env:GIFT_EVAL = (Resolve-Path .\data\GiftEval).Path
python -m experiments.run_quantization_pilot --config configs/quantization_milestone3.yaml
python -m experiments.compare_quantization_results --config configs/quantization_milestone3.yaml
python -m experiments.plot_quantization_impact `
  --comparison-root results/milestone3/comparisons `
  --dataset saugeenday/D/short --window-id 0
```

The first two datasets reuse identical Milestone 2 FP32/BF16/INT8 artifacts;
the M3 config does not require unnecessary reruns. The optional bounded probe
interface is explicit and writes outside the full comparison root:

```powershell
python -m experiments.run_quantization_pilot `
  --config configs/quantization_milestone3.yaml `
  --dataset bitbrains_fast_storage/5T/short `
  --dataset car_parts_with_missing/M/short `
  --max-windows 1 --output-root results/milestone3_probes --no-subprocess
```

Bounded probes are not official full-dataset aggregates and are marked
`evaluation_scope=bounded_probe`.

## Artifact layout

Each full run contains:

```text
results/milestone3/<precision>/<dataset>/run_###/
├── config.json
├── environment.json
├── metrics.json
├── forecasts.parquet
├── contexts.parquet
├── per_window_metrics.csv
└── summary.json
```

Forecast rows retain dataset, series, window, time step, ground truth, point
forecast, and every q-level. Context rows preserve the exact history used for
each forecast. Generated full M3 results are ignored by Git because downloaded
data and model outputs are local artifacts.

## Current Status

### Milestone 3 — real INT4 characterization

Milestone 3 completed a genuine INT4 TimesFM 2.5 inference run on
`saugeenday/D/short`. It used 20 official windows, horizon 30, context length
1024, one effective series, point shape `(20, 30)`, quantile shape
`(20, 30, 9)`, and q10--q90 levels. The saved run was reloaded in a fresh
process; finite outputs, ordered quantiles, context/ground-truth alignment,
and local plus official diagnostics were reproduced without rerunning
TimesFM.

The exact successful INT4 condition was TorchAO
`PrototypeInt4WeightOnlyConfig(group_size=128)`: CPU tinygemm A16W4,
asymmetric per-group INT4 weights, FP32 activations, group size 128. It
quantized 87 of 89 Linear modules. The unquantized Linear modules were
`tokenizer.hidden_layer` and `tokenizer.residual_layer`; the complete module
lists and excluded non-Linear types are saved in `metrics.json`.

The active runtime was Windows 10 build 26200, Python 3.11, PyTorch
`2.12.1+cpu`, TorchAO `0.18.0`, TimesFM `2.0.2`, on a 28-thread Intel CPU.
An RTX 4060 Laptop GPU was present, but no CUDA PyTorch runtime was active;
therefore no GPU speedup or GPU memory result is claimed. Stable TorchAO
INT4 was also tested and failed at the real TimesFM module with
`Requires mslk >= 1.0.0`; the official prototype CPU path was used instead.

#### Full official-window results

All values below come from generated artifacts. WQL is the official
`mean_weighted_sum_quantile_loss`; MASE is official `MASE[0.5]`. Coverage and
width are q10--q90 (nominal 80%). Runtime is forecast runtime in seconds.

| Dataset | Precision | Official MASE | Official WQL | Local MASE | MAE | RMSE | Pinball | Coverage80 | Width80 | Storage (MiB) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| saugeenday/D/short | FP32 | 0.957658 | 0.399888 | 0.957658 | 14.402810 | 37.793442 | 6.172970 | 0.786667 | 31.963772 | 882.299 | 15.058 |
| saugeenday/D/short | BF16 | 0.958503 | 0.399844 | 0.958502 | 14.415568 | 37.811287 | 6.172290 | 0.786667 | 32.012327 | 441.149 | 36.157 |
| saugeenday/D/short | INT8 | 0.957529 | 0.400070 | 0.957529 | 14.400769 | 37.804459 | 6.175782 | 0.770000 | 31.995339 | 221.765 | 23.625 |
| saugeenday/D/short | INT4 | 0.941699 | 0.404470 | 0.941699 | 14.162478 | 37.390130 | 6.243707 | 0.850000 | 35.577992 | 111.205 | 187.092 |
| bizitobs_l2c/5T/short | FP32 | 0.283652 | 0.078124 | 0.283652 | 2.738579 | 4.701967 | 1.137366 | 0.749107 | 7.852345 | 882.299 | 69.337 |
| bizitobs_l2c/5T/short | BF16 | 0.283834 | 0.078147 | 0.283834 | 2.739022 | 4.695142 | 1.137709 | 0.749256 | 7.862237 | 441.149 | 206.039 |
| bizitobs_l2c/5T/short | INT8 | 0.283545 | 0.078100 | 0.283545 | 2.738276 | 4.695800 | 1.137027 | 0.751935 | 7.896858 | 221.765 | 168.722 |
| bizitobs_l2c/5T/short | INT4 | not executed full | — | — | — | — | — | — | — | — | — |

#### Full-dataset coverage status

| Configuration | Full official windows | Status |
|---|---:|---|
| `saugeenday/D/short` | 20 | FP32/BF16/INT8 reused; real INT4 executed |
| `bizitobs_l2c/5T/short` | 140 | FP32/BF16/INT8 reused; INT4 not completed on CPU |
| `bitbrains_fast_storage/5T/short` | 45,000 | Not completed; real one-window probe for all four precisions |
| `car_parts_with_missing/M/short` | 2,674 | Not completed; real one-window probe for all four precisions |

One INT4 batch of 20 Saugeenday windows took 188.8 seconds on CPU. The two
large configurations would require many hours to multiple days at this
measured throughput. They are therefore represented as missing in the
automatically generated full-dataset table, not replaced by probe estimates.

#### INT4 effect and distortion

For Saugeenday, INT4 minus FP32 was ΔMASE `-0.015959` (relative `-1.666%`),
ΔWQL `+0.004582` (relative `+1.146%`), ΔCoverage80 `+6.333 percentage
points`, and ΔWidth80 `+3.614` (relative `+11.307%`). Mean absolute quantile
deviation was q50 `0.880983`, lower-tail q10--q20 `3.795647`, and upper-tail
q80--q90 `3.489523`; the mean q10--q90 width change was `+3.614219`.
Thus tail distortion was materially larger than median distortion in the
completed INT4 configuration, while aggregate point error did not increase.

The available INT8 deltas remain small: Saugeenday ΔWQL `+0.000182`,
ΔCoverage80 `-1.667 pp`, ΔWidth80 `+0.031566`; Bizitobs ΔWQL `-0.000023`,
ΔCoverage80 `+0.283 pp`, ΔWidth80 `+0.044513`. The complete absolute,
relative, coverage-point, per-window, and per-quantile values are generated
from artifacts rather than hard-coded in the comparison scripts.

#### Scientific assessment

**WEAK SIGNAL (provisional for the four-configuration scope).** The completed
real INT4 Saugeenday run shows clear quantile/tail distortion and an 11.3%
interval-width increase, with official probabilistic degradation larger than
the point change. However, full INT4 results are unavailable for the other
three selected configurations on this CPU environment, and BF16/INT8 remain
close to FP32. The evidence is therefore insufficient for GO or STRONG GO;
the next milestone must not start the proposed method until the remaining
full INT4 coverage is reviewed.

## Tables and figures

M3 tables are automatically generated as CSV, Markdown, and LaTeX:

- [quantization summary](results/milestone3/comparisons/quantization_summary.csv)
- [quantization deltas](results/milestone3/comparisons/quantization_deltas.csv)
- [quantile distortion](results/milestone3/comparisons/quantile_distortion.csv)
- `alignment_validation.json`

Publication-ready Figure A (impact), Figure B (quantile distortion), Figure C
(point versus probabilistic degradation), and Figure D (saved FP32/INT4
interval example) are written as PDF, SVG, and 300-dpi PNG under
`results/milestone3/comparisons/`.

## Contamination status

Dataset contamination risk and candidate final-paper configurations are
documented in [docs/gift_eval_selection.md](docs/gift_eval_selection.md).
Electricity remains development-only because TimesFM pretraining overlap
cannot be ruled out. No GIFT-Eval result in this repository is presented as
certified leakage-free.
