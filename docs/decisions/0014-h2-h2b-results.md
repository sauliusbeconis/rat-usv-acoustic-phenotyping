# 0014 — H2 (trajectory structure) and H2b (weight) results

Date: 2026-10-01. Run `.local/runs/20261001T152801.965331Z-h2-528d3234` (`usv h2`), using the 9 M4 runs. All geometry is for pups of the litter each fold's encoder did not see (28 pups); classical spaces are standardised with training-litter statistics.

## H2 metrics (pooled over held-out litters)
- **Distance–age ρ:** Spearman correlation, within each pup, between centroid distance and age gap (null: days shuffled, ≈ 0).
- **Alignment:** mean |age of littermate's nearest day − own age| (null ≈ 6.8 d).

| Space | ρ | Pups above null | Alignment (d) |
|---|---|---|---|
| Classical f0 / duration / call rate alone | 0.19 / 0.06 / 0.05 | 27 / 19 / 15 of 28 | 5.3 / 5.6 / 6.5 |
| Classical multivariate | 0.38 | 28/28 | 3.48 |
| VAE (seeds 0–2) | 0.31 / 0.34 / 0.25 | 28/28 | 3.14 / 3.29 / 3.83 |
| CNN (trained on age) | 0.80 / 0.75 / 0.80 | 28/28 | 1.62 / 1.67 / 1.85 |
| SSL | 0.31 / 0.33 / 0.30 | 28/28 | 3.38 / 3.12 / 3.40 |
| **Session control: background band powers** | **0.36** | 25/28 | **2.97** |
| Session control: each encoder's background-crop embeddings | −0.07 to 0.02 | 6–17/28 | 6.35–6.94 (≈ null) |

## Interpretation
1. Multivariate spaces (classical or learned) give far more consistent trajectories than any single classical variable. Development is not captured by one summary variable.
2. The unsupervised VAE space is **as** trajectory-consistent as the multivariate classical space, **not more**. The CNN's near one-dimensional, highly consistent geometry is expected from its age-regression objective, so it is not independent evidence of trajectory structure.
3. **Session confound:** within a litter, day = session = age, and call-free background band powers form an equally strong "trajectory" (ρ 0.36, alignment 2.97 d). Within-litter trajectory geometry is therefore not, by itself, evidence of development. The learned encoders do not respond to background content (their background embeddings show no trajectory), but session effects that act on the pups themselves (temperature, handling) cannot be excluded.

**H2 status: OBSERVED, confounded.** The cross-litter decoding (H1, decision 0013) remains the stronger evidence that call structure carries age information.

## H2b (acoustic age vs body weight)
Mixed model: (predicted − true age) ~ weight z-score within litter × day + age spline + sex, random intercept per pup. In all 9 runs and for classical, learned and combined predictions, the weight effect is ≈ 0 (−0.06 to +0.19 days per weight SD; every 95% CI includes 0, e.g. CNN learned seed 0: +0.11 [−0.12, 0.34]). **H2b: not supported**; heavier pups do not sound older for their age in this cohort.
