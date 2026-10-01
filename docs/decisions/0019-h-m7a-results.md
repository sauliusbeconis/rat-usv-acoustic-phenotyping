# 0019 — H-M7a results (pre-registered in decision 0018)

Date: 2026-10-01. Run `.local/runs/20261001T165708.850941Z-shape-test-fd6e9c44` (`usv shape-test`); shape features from features run `20261001T164535.599872Z-features-c17ba600`. Analysis exactly as pre-registered; post-hoc rows are labelled.

## Test A — this cohort (hypothesis-informed, not independent)
- **A1 (direction): supported, 3 of 3.** Per pup-day medians rise with age: U-index +0.021 kHz/day (p = 6×10⁻¹⁴), FM residual SD +0.030 kHz/day (p = 2×10⁻¹⁷), FM turning points +0.0066 per 10 ms per day (p = 2×10⁻¹⁵); 494 pup-days, 28 pups.
- **A2 (added information): supported.** Classical + shape vs classical, leave-one-litter-out: per-pup MAE 3.42 → 2.63 days, gain +0.79 [0.66, 0.93]; 28/28 pups improve.
- **A3 (explanation): 76% of the VAE gain is absorbed by shape features.** VAE gain over classical 0.74 / 0.66 / 0.59 days vs over classical + shape 0.19 / 0.15 / 0.14 (seeds 0/1/2); absorbed 74% / 78% / 77%. The residual VAE gain over classical + shape has a CI above 0 only for seed 0 ([0.06, 0.32]; seeds 1–2 [−0.01, 0.31], [−0.01, 0.28]).

## Test B — independent data (Fragile X rat pups, DeepSqueak table)
53 pup-days (files with ≥ 5 calls) from 33 rats.
- **B1 (WT, primary): not supported.** Base 2.79 → base + FM 2.67 days, gain +0.12 [−0.44, 0.64], 11/17 rats.
- B2 (KO): +0.20 [−0.40, 0.78], 9/16 rats.
- B3 (WT direction): median sinuosity vs age ρ = 0.06; modulated-label share vs age ρ = 0.31.
- *Post hoc*, excluding the P10-only cohort (age = cohort for those pups): WT (7 rats) −0.87 [−2.15, 0.24]; KO (10 rats) +1.01 [0.11, 1.94]. Contradictory and tiny.

## Conclusion
- **Within this cohort:** a small, interpretable set of pre-specified contour-shape features (onset/offset sweeps, curvature, fine FM) carries strong age information beyond conventional summaries and accounts for about three quarters of what the learned VAE embedding adds. This is the explanation chain's main result: the learned advantage is largely **contour shape**.
- **Independently:** not confirmed. The Fragile X test is inconclusive rather than negative: 25 WT pup-days, mostly one or two per rat, 5 sparse ages, P10 cohort confound, and only DeepSqueak's cruder FM proxies are available (no contours, so the U-shape component cannot be tested).
- **H-M7a status: OBSERVED and consistent within the cohort (pre-registered features); external replication pending, needs audio or contours.**
