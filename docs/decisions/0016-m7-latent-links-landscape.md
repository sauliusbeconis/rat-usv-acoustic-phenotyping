# 0016 — M7 part 2: latent ↔ properties, within-family signal, acoustic landscape

Date: 2026-10-01. Run `.local/runs/20261001T155358.010167Z-landscape-d7972289` (`usv landscape`; VAE seed 0, leave-one-litter-out folds).

## Method
In each fold, a call-level **latent age score** is a ridge regression (α = 10) of pup age on standardised VAE embeddings, fitted on the training litters. A ridge map from 9 classical call properties to that score is also fitted on the training litters. On held-out calls, its prediction is the "classical part" and the remainder is the **residual**, the latent age signal that classical properties do not explain.

## Results (held-out calls, mean over 3 folds)
- The latent age score correlates with age at ρ = 0.38 (single calls; animal-day pooling in M4 is much stronger).
- Classical call properties explain only **14%** of the latent score's variance (R² 0.20 / 0.17 / 0.06 per fold).
- **The residual still tracks age:** ρ = 0.27 (0.34 / 0.23 / 0.23), and within most VAE call families (positive in 7 of 8 families for E2L1 and E2L3, 6 of 8 for E2L2). So the extra information is not only repertoire composition (decision 0015): it is also within-family morphology.
- Latent dimensions correlate with classical properties in a distributed way (`latent_property_heatmap.png`); no single dimension is a simple proxy.

## Hypothesis generated (not tested; from inspecting held-out calls, so post hoc)
**H-M7a:** the learned age signal reflects **contour shape**, meaning onset and offset frequency sweeps (U-shaped calls read "older" at P10–P13) and fine frequency modulation or trill-like wobble (read "older" at P17–P20). Summary statistics such as median f0, slope and bandwidth miss these. Montages: `residual_montage_<litter>.png`. Because it was formed by looking at held-out litters, it must be tested on data not used here (e.g. M8, or a pre-registered contour-shape feature set evaluated on new data).

## Acoustic landscape (visualisation class 3: conceptual, data-derived)
`landscape_2d_<litter>.png` and `landscape_3d_<litter>.png`. Axes are the latent age score (fitted on other litters) and the main latent variation orthogonal to it (training-fitted). The terrain is the smoothed density of held-out calls; the black line with coloured points is the held-out litter's day-mean trajectory P1→P21; orange lines are individual pups (pup-days with ≥ 5 calls, 3-day rolling mean for display). Each figure caption carries the quantitative link (ρ of score and of residual with age). It is not evidence by itself: evidence is in decisions 0013–0016.

## Status
M7 first full pass done: families, typical calls, composition, latent–property links, residual examples, landscape and explorer. Not done: CNN-space versions, multi-seed landscape stability, and testing H-M7a.
