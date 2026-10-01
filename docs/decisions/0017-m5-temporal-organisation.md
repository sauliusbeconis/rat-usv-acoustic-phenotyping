# 0017 — M5 temporal organisation (H3): null result

Date: 2026-10-01. Owner design decision: hand-built features first; a sequence model only if call order clearly carries information. Runs: `.local/runs/20261001T162525.491013Z-temporal-62ae5330` (pre-specified full set) and `.local/runs/20261001T162711.228801Z-temporal-d582c430` (post-hoc compact set). Uses the three VAE M4 runs, leave-one-litter-out.

## Features (per pup-day, confident calls in time order)
- **Timing (order-free):** inter-call interval 10/50/90th percentiles and CV, silent-gap median, bouts (gap > 0.5 s starts a bout, fixed a priori): bouts per minute, calls per bout, median bout duration.
- **Sequence (order-dependent):** 8×8 transition probabilities between the fold's VAE call families (M7), conditional entropy H(next | current), same-family repeat rate, local variation (LV) of successive intervals.
- **Control:** sequence features recomputed after shuffling call order within each recording (5 shuffles); composition and interval distribution are preserved.

## Results (per-pup MAE difference, pup bootstrap 95% CI; positive = better)
| VAE seed | Timing over classical+VAE | Full sequence (67) over timing | Real vs shuffled order | Compact sequence (3, post hoc) over timing | Compact: real vs shuffled |
|---|---|---|---|---|---|
| 0 | +0.04 [−0.01, 0.10] | −0.09 [−0.21, 0.03] | +0.03 [−0.06, 0.12] | +0.01 [−0.01, 0.04] | 0.00 [−0.02, 0.03] |
| 1 | +0.05 [0.00, 0.11] | −0.14 [−0.24, −0.03] | −0.06 [−0.14, 0.02] | −0.02 [−0.06, 0.01] | 0.00 [−0.03, 0.03] |
| 2 | +0.01 [−0.07, 0.09] | −0.04 [−0.15, 0.07] | −0.09 [−0.19, 0.01] | −0.02 [−0.05, 0.01] | −0.03 [−0.05, 0.00] |

## Interpretation
- Order-free timing adds at most a marginal amount (CI > 0 in one of three seeds).
- **Call order carries no detectable age information** beyond single-call structure: real order never beats shuffled order. The pre-specified 67-feature sequence set slightly hurts, likely overfitting; the post-hoc compact set (entropy, repeat rate, rhythm LV) is flat.
- **H3: not supported** at the feature level in this cohort. Following the owner's rule, no sequence model (GRU) was trained. Limits: 5-min isolation recordings, families from an unstable 8-cluster split (decision 0015), ~380 training pup-days per fold. A sequence model on more data could still find order effects; this is not evidence of absence.
