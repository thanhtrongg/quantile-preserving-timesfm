# Milestone 3 low-bit backend and scope

Milestone 3 is an observation-only low-bit characterization. It does not
implement the proposed quantile-preserving PTQ method, custom quantile losses,
QAT, mixed-precision search, fine-tuning, pruning, distillation, or another
benchmark.

## Exact executed INT4 backend

The stable TorchAO `Int4WeightOnlyConfig(group_size=128)` path was first
tested against the real TimesFM module. It failed before inference because the
installed stable path requires `mslk >= 1.0.0`. The official TorchAO prototype
CPU path was then tested on a toy Linear and on TimesFM itself. Both produced
finite outputs from packed INT4 weights; no dtype cast, manual rounding, or
simulated dequantization was used.

The executed INT4 condition is:

| Field | Executed value |
|---|---|
| Backend | TorchAO `PrototypeInt4WeightOnlyConfig` |
| Kernel | CPU tinygemm A16W4 (`Int4OpaqueTensor`) |
| Weight precision | INT4, 4-bit packed weights |
| Activation precision | FP32 activations (A16W4 backend; “A16” denotes floating-point activations) |
| Granularity | asymmetric per-group weight quantization |
| Group size | 128 input features per group |
| Hardware | Windows 10 build 26200, Intel 28-thread CPU; PyTorch CUDA unavailable |
| Torch compile | disabled because the Windows environment has no `cl.exe` |

The TimesFM model contains 89 Linear modules. The successful INT4 run
quantized 87 of them. The two unquantized Linear modules were
`tokenizer.hidden_layer` and `tokenizer.residual_layer`; these did not satisfy
the prototype kernel's compatible weight shape. Non-Linear modules were also
excluded: `ModuleList`, `MultiHeadAttention`, `PerDimScale`, `RMSNorm`,
`ResidualBlock`, `RotaryPositionalEmbedding`, `SiLU`, and `Transformer`.
The complete module list is stored in each INT4 run's `metrics.json` and
`environment.json`.

The official TorchAO prototype source documents this CPU A16W4 path as
supporting floating-point activations and groupwise INT4 weights. The stable
TorchAO path and its MSLK dependency remain documented as an attempted,
non-executed alternative. An RTX 4060 Laptop GPU is present (driver 591.74,
reported CUDA 13.1, 8188 MiB), but the active Python environment is
`torch==2.12.1+cpu`; therefore no GPU metric is claimed. The attempted CUDA
wheel installation was not completed because the system volume had only about
1.5 GiB free during wheel extraction. This is an environment limitation, not
an INT4 simulation.

## Other executed conditions

| Variant | Backend | Weight representation | Activation representation | Granularity |
|---|---|---|---|---|
| FP32 | official TimesFM PyTorch | FP32 | FP32 | n/a |
| BF16 | PyTorch BF16 + CPU autocast | BF16 | BF16 autocast kernels | n/a |
| INT8 | TorchAO `Int8WeightOnlyConfig` | symmetric INT8 | FP32 | per output channel/row |
| INT4 | TorchAO prototype CPU A16W4 | asymmetric packed INT4 | FP32 | per group, 128 |

All variants use the same TimesFM 2.5 checkpoint, context length 1024,
official GIFT-Eval windows, horizons, ground truth, and q10--q90 levels
`[0.1, 0.2, ..., 0.9]`. The continuous head returns point/mean plus nine
quantiles; artifacts store point forecasts as `(B, H)` and quantiles as
`(B, H, 9)`.

## M3 coverage and resource limitation

Full official-window artifacts were available for `saugeenday/D/short` and
`bizitobs_l2c/5T/short`. Existing FP32/BF16/INT8 artifacts were reused when
their configurations were identical to M2. A real one-window probe was also
run for every precision on Bitbrains and Car Parts to verify loader, horizon,
shape, and INT4 compatibility; these probes are explicitly marked
`evaluation_scope=bounded_probe` and are not included in full-dataset tables.

The official downloaded snapshot resolves to 45,000 Bitbrains windows
(horizon 48) and 2,674 Car Parts windows (horizon 12). On this CPU, one real
INT4 batch of 20 Saugeenday windows took 188.8 seconds, while the saved full
INT4 run took 177.5 forecast seconds for 20 windows. Extrapolating that
measured throughput would require many hours for Car Parts and multiple days
for Bitbrains, before the other precisions. The repository therefore reports
those two full configurations as `missing` rather than fabricating aggregate
metrics or relabelling bounded probes as official GIFT-Eval results.

This limitation prevents a four-configuration generalization claim. It does
not invalidate the completed real INT4 result or its saved-artifact checks.

## Artifact discipline

Every full precision run writes the same identifiers to Parquet:
`dataset`, `series_id`, `forecast_window_id`, and `time_step`. The comparison
script verifies forecast identifiers, ground truth, and contexts before
computing quantile distortion. Per-window diagnostics remain in each run's
`per_window_metrics.csv`; forecast-step and series-level rows are retained in
`forecasts.parquet`.

The full M3 comparison outputs are generated from artifacts:

- `results/milestone3/comparisons/quantization_summary.csv` (also Markdown/LaTeX)
- `results/milestone3/comparisons/quantization_deltas.csv` (also Markdown/LaTeX)
- `results/milestone3/comparisons/quantile_distortion.csv` (also Markdown/LaTeX)
- `results/milestone3/comparisons/alignment_validation.json`

Fresh-process verification loads the saved Parquet files, checks finite and
ordered quantiles, checks context/ground-truth alignment against the official
loader, and reproduces local diagnostics without loading TimesFM or rerunning
inference. The INT4 Saugeenday verification also reproduced official
`MASE[0.5]` and official mean weighted quantile loss from the saved forecasts.
