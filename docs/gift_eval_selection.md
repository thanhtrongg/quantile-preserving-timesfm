# GIFT-Eval pilot selection

Milestone 1 uses the official SalesforceAIResearch GIFT-Eval loader and its
official test protocol. The loader creates the final 10% test split and
non-overlapping rolling windows; this repository does not shuffle series,
construct a replacement split, or edit ground truth.

The pilot deliberately spans several domains and frequencies while keeping the
short term configuration. `# Series` is resolved from the downloaded official
Arrow dataset at runtime. This is intentional: `to_univariate: true` follows
GIFT-Eval's supported multivariate-to-univariate compatibility path, so the
effective count depends on the exact dataset snapshot and can differ from the
raw configuration's number of variates.

| Dataset/Config | Domain | Frequency | Horizon | # Series | Probabilistic Evaluation Support | Why Selected | Leakage/Contamination Risk |
|---|---|---:|---:|---:|---|---|---|
| `electricity/15T/short` | Energy | 15 min | 48 | runtime-resolved | Yes | Medium-sized high-frequency energy series and a canonical GIFT-Eval configuration | Development-only: high contamination concern; do not present as leakage-free |
| `solar/10T/short` | Energy | 10 min | 48 | runtime-resolved | Yes | Different energy signal and sampling cadence from Electricity | Possible benchmark overlap; not treated as leakage-free |
| `hospital/short` | Healthcare | Monthly | 12 | runtime-resolved | Yes | Monthly healthcare demand, giving a lower-frequency domain contrast | Possible indirect overlap through documented GiftEvalPretrain provenance; no zero-leakage claim |
| `restaurant/short` | Sales | Daily | 30 | runtime-resolved | Yes | Daily sales series with a clear short horizon | Possible indirect overlap through documented GiftEvalPretrain provenance; no zero-leakage claim |
| `bitbrains_fast_storage/5T/short` | Web/CloudOps | 5 min | 48 | runtime-resolved | Yes | High-frequency operations data from a distinct domain | Lower relative risk candidate; possible indirect GiftEvalPretrain overlap remains unverified |

The uncapped Milestone 1 validation uses `saugeenday/D/short` (daily,
horizon 30, one series, 20 official windows). It is a lower-relative-risk
pilot rather than a claim of zero contamination. The public [TimesFM 2.5 model
card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) lists
`GiftEvalPretrain` among pretraining sources, so every GIFT-Eval
test result is treated as potentially contaminated unless a direct overlap
audit establishes otherwise. Electricity is retained for pipeline development
only.

Candidate configurations for the final paper, subject to a documented overlap
audit, are:

- `bitbrains_fast_storage/5T/short` (CloudOps, 5-minute horizon 48)
- `bizitobs_l2c/5T/short` (IT operations, 5-minute horizon 48)
- `saugeenday/D/short` (daily horizon 30)
- `car_parts_with_missing/M/short` (monthly horizon 12)

These are lower-relative-risk choices because their public names are not
directly identified as TimesFM 2.5 pretraining sources in the model card. They
are not certified leakage-free: shared `GiftEvalPretrain` provenance and the
benchmark's historical leakage concerns must still be reported.

The domain, frequency, variate counts, and horizon mapping are taken from the
official GIFT-Eval metadata/loader. The loader's current mapping gives 48 for
the `T` configurations, 30 for the daily `D` configurations, and 12 for the
monthly `M` configuration. If one of the
paths is absent in a downloaded release, the runner fails with the exact path;
it does not silently substitute another dataset.

The official evaluator's primary probabilistic result is retained under its
native metric name. Local diagnostics additionally record mean pinball loss,
80% interval coverage/width when q10 and q90 exist, and quantile crossing rate.
TimesFM 2.5's documented output provides q10 through q90, so a 90% interval
(q05--q95) is not fabricated and is stored as unavailable.

## Milestone 2 executed configurations

The low-bit pilot used two lower-relative-risk configurations rather than
Electricity. `saugeenday/D/short` resolved to one effective univariate series
and 20 official windows (horizon 30). `bizitobs_l2c/5T/short` resolved to seven
effective univariate series and 140 windows in total (horizon 48). Both used a
configured context length of 1024 and the same q10--q90 output levels for every
precision variant.

`bitbrains_fast_storage/5T/short` remains a candidate for the final paper,
but its current `to_univariate` expansion produces 45,000 windows. It was
therefore not used as primary CPU evidence in Milestone 2. This is a workload
scope decision, not a contamination clearance; its status remains lower
relative risk but unverified for direct overlap.

## Milestone 3 execution status

Milestone 3 kept the four selected configurations unchanged. Their
contamination status is unchanged as well: all remain potentially contaminated
because the TimesFM 2.5 model card lists `GiftEvalPretrain` among pretraining
sources; none is certified leakage-free.

| Configuration | Resolved official windows | Contamination status | M3 execution status |
|---|---:|---|---|
| `saugeenday/D/short` | 20 | Lower-relative-risk candidate; direct overlap unverified | Full FP32/BF16/INT8 reuse and real INT4 |
| `bizitobs_l2c/5T/short` | 140 | Lower-relative-risk candidate; direct overlap unverified | Full FP32/BF16/INT8 reuse and real INT4 |
| `bitbrains_fast_storage/5T/short` | 45,000 | Lower-relative-risk candidate; direct overlap unverified | One-window real probe; full low-bit matrix resource-limited |
| `car_parts_with_missing/M/short` | 2,674 | Lower-relative-risk candidate; direct overlap unverified | One-window real probe; full low-bit matrix resource-limited |

The downloaded official Car Parts snapshot is stored at
`car_parts_with_missing` and reports monthly frequency `M`; the public
configuration label remains `car_parts_with_missing/M/short`. Bounded probes
are retained separately under `results/milestone3_probes/` and are not used as
official aggregate metrics. Electricity remains development-only and was not
used for the M3 matrix.
