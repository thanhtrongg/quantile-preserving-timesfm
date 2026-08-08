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
| `electricity/15T/short` | Energy | 15 min | 48 | runtime-resolved | Yes | Medium-sized high-frequency energy series and a canonical GIFT-Eval configuration | TimesFM is zero-shot here; inspect the benchmark/model contamination records before reporting as zero-leakage |
| `solar/10T/short` | Energy | 10 min | 48 | runtime-resolved | Yes | Different energy signal and sampling cadence from Electricity | Same benchmark-level review required |
| `hospital/M/short` | Healthcare | Monthly | 12 | runtime-resolved | Yes | Monthly healthcare demand, giving a lower-frequency domain contrast | Same benchmark-level review required |
| `restaurant/D/short` | Sales | Daily | 30 | runtime-resolved | Yes | Daily sales series with a clear short horizon | Same benchmark-level review required |
| `bitbrains - fast storage/5T/short` | Web/CloudOps | 5 min | 48 | runtime-resolved | Yes | High-frequency operations data from a distinct domain | Same benchmark-level review required |

The domain, frequency, variate counts, and horizon mapping are taken from the
official GIFT-Eval metadata/loader. The loader's current mapping gives 48 for
the `T` configurations and 12 for the monthly `M` configuration. If one of the
paths is absent in a downloaded release, the runner fails with the exact path;
it does not silently substitute another dataset.

The official evaluator's primary probabilistic result is retained under its
native metric name. Local diagnostics additionally record mean pinball loss,
80% interval coverage/width when q10 and q90 exist, and quantile crossing rate.
TimesFM 2.5's documented output provides q10 through q90, so a 90% interval
(q05--q95) is not fabricated and is stored as unavailable.
