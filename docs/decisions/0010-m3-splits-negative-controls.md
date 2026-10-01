# 0010 — M3 frozen splits and negative controls

Date: 2026-10-01. Run: `.local/runs/20261001T122013.275165Z-controls-c4cf4161` (`usv controls`, 100 permutations).

## Splits (frozen)
`splits.json` in the run, SHA-256 `67b01844307763c237eca3ef1b43231c0abdac392f5c201209ba6a0483313709`. Leave-one-animal-out (28 folds) and leave-one-litter-out (3 folds); every animal is tested exactly once. Test animals are never used for imputation, scaling or tuning (inner `GroupKFold` by animal).

## Controls (animal-day level; fixed decoders: median imputation + missingness indicators, standardisation, ridge with inner-CV alpha; per-animal MAE with 95% bootstrap CI over animals)
| Input | Leave-one-animal-out | Leave-one-litter-out |
|---|---|---|
| Classical M2 features | MAE 3.35 d [3.13, 3.59], ρ 0.72; null 5.24, p = 0.01 | MAE 3.42 d, ρ 0.71; litter bias +0.45 / +0.87 / −0.66 d |
| Background-only (14 band powers of call-free windows) | MAE 4.02 d [3.52, 4.51], ρ 0.53; p = 0.01 | MAE 5.88 d, ρ −0.14 |
| SNR/amplitude-only | MAE 3.98 d [3.60, 4.35], ρ 0.60; p = 0.01 | MAE 5.50 d, ρ 0.04 |

The null shuffles age within each animal; p = 0.01 is the minimum possible with 100 permutations.

**Discovered confound: a session leak in leave-one-animal-out.** Background audio and SNR "decode" age only when the held-out pup's littermates, recorded in the same daily session, are in the training set. They fail completely across litters. So recording-session conditions identify the day, and the day gives the age. The classical features keep their accuracy across litters (3.42 vs 3.35 d), so their age signal is not explained by session conditions. **Consequence for M4:** leave-one-animal-out alone is not a valid primary test for learned representations, which can pick up session acoustics. Any claim must also hold under leave-one-litter-out, or under a session-blocked scheme (see the owner decision in the roadmap).

**Calendar date versus age** (mixed models, sex as covariate, litter omitted because date = age + litter offset): age beyond date is highly significant for all outcomes tested (χ² 66–177, p ≤ 10⁻¹³). Date beyond age is also significant for f0 (χ² 19.0, p = 3×10⁻⁴) and duration (χ² 24.9, p = 2×10⁻⁵), weak for call rate (p = 0.048) and absent for tonality (p = 0.23). With 3 litters, a "date" effect cannot be separated from litter differences; it shows that litters differ beyond age for f0 and duration.

**Identity decoding** (train on even days, test on odd, and swap; fixed logistic regression): animal 0.084 vs chance 0.036 (28 classes); within litters 0.16/0.10, 0.33/0.25, 0.17/0.07; litter 0.55 vs 0.33. Individual identity is weak in classical features, while litter (or batch) is detectable.

## Status
Classical age trajectories: OBSERVED → INTERNALLY_VALIDATED at the level "age information in classical features generalises to held-out animals and held-out litters, and is not explained by background or SNR". Developmental versus cumulative-handling or session effects remain inseparable within this design.
