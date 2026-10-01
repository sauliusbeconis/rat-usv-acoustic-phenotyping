# 0015 — M7 explanation chain (first pass)

Date: 2026-10-01. Runs: `usv explain` on the VAE seed 0/1/2 M4 runs (`.local/runs/20261001T153401.888949Z-explain-1a8315ae` and two later explain runs), explorer `.local/runs/20261001T153507.631560Z-explorer-419c4a5c/explorer.html`.

## Method
- VAE call embeddings, per leave-one-litter-out fold, are clustered into K = 8 call families (fixed a priori) by k-means fitted on the training litters and applied to the held-out litter.
- For each family: classical acoustic profile, share of calls by postnatal day, and the 8 held-out calls nearest the family centre (montage).
- Explanatory test: per-day family proportions are added to the classical features under the same leave-one-litter-out decoder as M4.

## Results
- **Families are acoustically interpretable** (held-out E2L3, seed 0): short (~40 ms) ~43 kHz calls dominate early and decline; long (140–190 ms) flat harmonic calls at 35–43 kHz are prominent mid-period; frequency-modulated or two-note calls rise after ~P8; very short (~28 ms), high (~61 kHz), upward-sloping calls grow after P12. Figures: `family_montage_<litter>.png`, `family_composition_<litter>.png`.
- **Repertoire composition adds age information beyond classical features** in every seed: per-pup MAE 3.42 → 3.19 / 2.92 / 3.13 days (gain +0.24 [0.13, 0.34], +0.51 [0.38, 0.63], +0.29 [0.20, 0.39]).
- **Composition explains only part of the VAE's gain, unstably:** 32%, 77% and 50% for seeds 0/1/2. The extra information in learned embeddings is partly *which call types are produced* and partly *within-type morphology*. The split depends on the clustering and is not stable enough to report as a single number.

## Explorer (visualisation class 2: exploratory)
A private HTML page (300 held-out E2L3 calls, stratified by age) plots the fold's VAE space in 2-D PCA fitted on the training litters. Clicking a point shows the call's spectrogram and plays the original audio 16× slowed (≈ 2.5 kHz for a 40 kHz call). The 2-D view shows only a weak age gradient, which is consistent with age information being spread over many latent dimensions. It is not evidence.

## Status and next steps
Explanatory, not confirmatory. Still to do for M7: link latent directions to measurable properties (e.g. per-dimension correlation with classical call features), show within-family morphology change with age, and produce the conceptual "acoustic landscape" figure (class 3), tied to the quantitative results above.
