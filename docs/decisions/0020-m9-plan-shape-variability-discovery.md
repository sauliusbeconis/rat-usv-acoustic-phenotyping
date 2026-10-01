# 0020 — M9 plan: data-driven discovery of developmental call structure (to start 2026-10-02)

Status: **EXECUTED 2026-10-01**; results and deviations in decision 0021. (Owner first approved the scope as "plan it, don't start", then asked to start the same evening.)

## Why
H-M7a (decisions 0018/0019) showed that six hand-picked contour-shape measures absorb ~76% of the VAE's age information. About a quarter is unexplained, and the six measures came from inspecting example calls. M9 replaces guessing with data-driven discovery and protects against false discoveries with a discovery/confirmation design.

## Questions
- **Q1 Shape vocabulary:** which data-derived contour shape modes change with age, and do they explain more than the six H-M7a measures?
- **Q2 Beyond the contour:** do the loudness envelope, harmonic structure, within-call tonality and nonlinear phenomena change with age and add information?
- **Q3 Variability and stereotypy:** does call-to-call variability shrink with age (overall and within call types)? This is a candidate marker of vocal motor maturation.
- **Q4 What remains:** which spectrogram regions carry the VAE age signal that classical + shape features miss?
- **Q5 Biological contrasts:** sex differences in trajectories; do individual vocal signatures strengthen with age?

## Safeguard: rotating discovery/confirmation design
For each litter L in {E2L1, E2L2, E2L3}:
1. **Discovery** on the other two litters only. Any code path that fits, ranks or selects features loads only discovery-litter rows; a guard asserts that L is absent.
2. **Lock** a predictions file `m9_predictions_holdout_<L>.json`: the selected features, the predicted sign of the age trend, and the predicted decoding gain direction. Record its SHA-256 in the run before any confirmation code runs.
3. **Confirmation** on L: the predictions are tested as locked, with no re-selection.

A finding counts as **confirmed** if it replicates in ≥ 2 of 3 rotations with the predicted sign, after Benjamini–Hochberg FDR at q = 0.05 across all features tested in that rotation. E2L2 has only 4 pups, so its rotation is reported but weighted as weak. Everything else is labelled exploratory.

Unit and statistics as protocol v1: pup-day aggregates; mixed models with a random intercept per pup for age trends; the same ridge decoder for information gain with a pup-level bootstrap.

## Work packages and order
| # | Package | Output | Est. time |
|---|---|---|---|
| 0 | **Contour store**: re-run feature extraction, also saving per-call time-normalised curves (32 points): f0 contour, amplitude envelope, harmonic-to-fundamental energy ratio, frame tonality; plus raw-length contour arrays. Read-only on sources. | `call_curves.npz` + index | ~15 min (one extraction run) |
| 1 | **Functional PCA** of time-normalised f0 contours (fit per rotation on discovery litters), shape modes plotted as mean ± k·SD curves; also fPCA of amplitude envelopes | modes, per-call scores, mode figures | ~1 h |
| 2 | **Beyond-contour features**: envelope rise/fall time, amplitude-modulation depth, harmonic ratio (where 2·f0 is observable), tonality profile (start/middle/end), nonlinear phenomena counts (jumps, subharmonic bands at f0/2, biphonation-like non-harmonic second bands) | per-call table | ~1 h |
| 3 | **Variability**: per pup-day dispersion (robust SD or MAD) of f0, duration and fPCA scores, overall and within the M7 call family; age trend + sex interaction | variability table, figures | ~45 min |
| 4 | **Discovery/confirmation runner**: rotation loop, locked predictions with digest, FDR, confirmation table | `m9_confirmation.json` | ~1 h |
| 5 | **Residual attribution (Q4)**: occlusion maps of the VAE age score on call images (start/middle/end × fundamental/harmonic bands), averaged per family and age band; latent traversal images along the age direction at fixed family | maps, montages | ~1 h |
| 6 | **Biological contrasts (Q5)**: sex × age interactions on confirmed features; individual-identity decodability by age band (train/test on different days within band) | tables, figures | ~45 min |
| 7 | Decision note (results), roadmap update, figure registry entries; **private results page** `private/results/M9_findings.html` with a figure and a plain-language explanation per finding (owner request) | docs, report | ~45 min |

Total: about one working day; GPU only for package 5 (minutes).

## Decision rules fixed now
- fPCA: keep modes explaining ≥ 90% cumulative variance (max 8). Curves are time-normalised to 32 points; duration is kept as a separate feature.
- Variability measure: MAD-based SD; pup-days need ≥ 10 confident calls.
- Subharmonic: an energy peak at f0/2 (±1 kHz) ≥ 6 dB above the local median in ≥ 3 consecutive frames. Biphonation-like: a second tonal band not within ±2 kHz of k·f0 for k ∈ {0.5, 1, 2}, present ≥ 3 frames.
- Information-gain tests compare against **classical + the six H-M7a shape features** (the current best interpretable baseline), not classical alone.

## Not in scope
New learned architectures (still owner-gated), NASS, external data (blocked on obtaining audio).
