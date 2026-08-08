# Milestone 2 quantization backend and scope

Milestone 2 is an observation-only precision pilot. It does not implement the
proposed quantile-preserving PTQ method, custom quantile losses, QAT, mixed
precision search, fine-tuning, distillation, or any other method contribution.

## Architecture inspection

The installed official TimesFM package is `2.0.2` and the checkpoint is
`google/timesfm-2.5-200m-pytorch`. The PyTorch module exposes 20 Transformer
blocks, 16 attention heads, model dimension 1280, fused QKV projections, RMSNorm,
rotary position embeddings, SiLU feed-forward blocks, and a continuous
quantile output head. The runtime model contains 89 `torch.nn.Linear` modules:

- tokenizer residual block: 3 Linear modules;
- 20 Transformer blocks: 4 Linear modules per block (fused QKV, attention
  output, and two feed-forward projections);
- point output projection: 3 Linear modules;
- continuous quantile output projection: 3 Linear modules.

The INT8 pilot quantizes all 89 Linear modules. RMSNorm, rotary embeddings,
SiLU, residual additions, attention/cache tensors, and other non-Linear
operations remain floating point. Quantile post-processing and TimesFM's
configured quantile-crossing fix are unchanged across precisions.

## Executed schemes

| Variant | Backend | Weight representation | Activation representation | Granularity | Inference status |
|---|---|---|---|---|---|
| FP32 | official TimesFM PyTorch | FP32 | FP32 | n/a | Executed |
| BF16 | PyTorch BF16 weights + CPU autocast | BF16 | BF16 autocast kernels | n/a | Executed |
| INT8 | TorchAO `Int8WeightOnlyConfig` | INT8, symmetric, per-row/per-output-channel | FP32 | per-row | Executed |
| INT4 | TorchAO `Int4WeightOnlyConfig(group_size=128)` | INT4 groupwise | floating-point dequantized compute according to backend | group size 128 | Unsupported on this CPU environment |

TorchAO INT8 is actual weight-only quantized inference; it is not a tensor
cast. The INT4 compatibility check was performed against the real TimesFM
module and failed before forecasting with `ImportError: Requires mslk >= 1.0.0`.
The HQQ/tile-packed alternative requires CUDA on this installation. No INT4
forecast numbers are reported.

The quantized runs use `torch_compile=False` because the Windows environment
does not provide `cl.exe`; the attempted TorchInductor path failed with that
compiler error. This is recorded as a runtime limitation. CPU runtime and
memory numbers therefore describe this eager CPU setup only.

## Artifact discipline

Every precision writes the same identifiers to Parquet:
`dataset`, `series_id`, `forecast_window_id`, and `time_step`. The comparison
script verifies that ground truth and context arrays are byte-equivalent across
FP32, BF16, and INT8 before computing distortion. Per-window metrics remain in
each run's `per_window_metrics.csv` for later bootstrap/confidence analysis.

The primary comparison outputs are generated from artifacts:

- `results/milestone2/comparisons/quantization_summary.csv`
- `results/milestone2/comparisons/quantization_summary.md`
- `results/milestone2/comparisons/quantization_summary.tex`
- `results/milestone2/comparisons/quantile_distortion.csv`
- `results/milestone2/comparisons/quantization_deltas.csv`
- `results/milestone2/comparisons/alignment_validation.json`
