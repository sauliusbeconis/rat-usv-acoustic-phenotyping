# 0021 — M9 results: data-driven discovery of developmental call structure

Status: accepted. Date: 2026-10-01 (started a day early at the owner's request). Plan: decision 0020.

## Runs (all `.local/runs/`)
- Curve store (WP0): `20261001T175036.684944Z-features-7f6e6304`. `call_features.parquet` is byte-identical to the H-M7a features run `…c17ba600` (regression check). Adds `call_curves.npz` (32-point f0, envelope, harmonic ratio, tonality per call; raw contours) and `call_curves_index.parquet` (nonlinear-phenomena counts).
- Discovery/confirmation + sex + identity (WP1–4, 6): `20261001T181436.048902Z-m9-5a678459` (`usv m9`). Locks in `m9_locks.txt`: E2L1 `bd79dbb4…`, E2L2 `5ba276e3…`, E2L3 `9c09ec26…`.
- Residual attribution (WP5, exploratory): `20261001T181732.397771Z-m9-attribution-bcb0fb9a` (`usv m9-attribution`, VAE seed 0).
- Failed runs, kept: `…174435…-features-153384ab` (crash: f0 above band), `…181236…`, `…181254…`, `…181313…`, `…181401…` (m9 crashes, see the deviation log). None of the crashes produced confirmation outcomes, and the discovery locks were reproduced bit-for-bit between reruns before the fallback fix. That fix added a `model` field to the trend records, which changed the lock digests.
- Results page (private): `private/results/M9_findings.html`. Figures are in `private/results/M9/`.

## Implementation choices and deviations from 0020 (logged as taken)
1. **Biphonation rule.** As written in 0020 (exclude ±2 kHz around 0.5, 1 and 2·f0), it flagged the shoulder of the main f0 band in 16 of 70 calls of a test recording. Now the whole above-threshold lobe around every multiple of f0/2 is excluded, and the band must be continuous (≤ 2 kHz per frame) for ≥ 3 frames. The detector is not validated against hand labels.
2. **Within-family variability** uses rotation-specific families: k-means, k = 8, on log duration, f0 median, bandwidth and the first three f0 modes, fitted on discovery litters. The M7 VAE families were not used: for discovery litters, those labels come from VAEs that saw the held-out litter's images.
3. **fPCA** runs on per-call mean-removed curves (shape only; the f0 level is in the baseline). Modes are named by Hungarian matching to discrete Legendre templates (data-free), so names are comparable across rotations. The QR-orthonormalised templates have an arbitrary but fixed sign, so read directions from the mode figures. Positive f0 **P1** = rising sweep; positive **P2** = arch (inverted U); positive envelope **P1** = loudness building toward the call end.
4. **Gain tests** are per feature block (f0 modes, envelope modes, beyond-contour, variability, all selected). The predicted direction is the sign of a two-litter inner check on discovery data. A gain counts as confirmed if the held-out CI excludes 0 in the predicted direction. No FDR is applied across the five block tests: there are few of them and they are correlated.
5. **Age-trend fallbacks** (added after crashes in the E2L2 confirmation fit, before any confirmation output existed): a constant feature gets p = NaN and is never significant. A singular mixed model falls back to OLS with pup-clustered SEs. BH runs over the finite p-values only.
6. Inner decoder tuning folds are capped at the number of training pups. E2L2 alone has 4, and the pipeline crashed on it before this. Results for ≥ 5 pups are unchanged.
7. **Sex × age table**: the rotation that held out E2L2 supplies the fPCA basis. That rotation fits on 24 pups, the largest discovery set.
8. **Identity**: per-call LDA on classical + H-M7a call features, leave-one-day-out within litter × age band, ≤ 30 calls per pup-day, pup-days with ≥ 10 calls. The null permutes pup labels among the pups recorded on the same day (100 permutations). The score is balanced accuracy.

## Results
**Q1 Shape vocabulary: stable, and it adds nothing beyond H-M7a.** Six f0 modes reach the 90% variance target (50/20/10/5/3/2%). Their shapes are nearly identical whichever litter is held out (fig. `modes_f0.png`), so the shape vocabulary is a property of the cohort and not of a particular litter. Age trends of the modes confirmed in 3 of 3 litters: P1 rises (calls sweep upward more with age), P2 falls (fewer arches, more U shapes; ρ = −0.95 with H-M7a curvature, so this is a data-driven replication of H-M7a's direction), and P3 falls. **Decoding gain of f0 modes over classical + H-M7a shape: none** (−0.16 [−0.40, 0.02], +0.03 [−0.18, 0.23], −0.02 [−0.07, 0.02] days). The six pre-registered shape measures already capture the age-relevant contour shape.

**Q2 Beyond the contour: robust age trends; added information promising but not confirmed.**
- Confirmed 3 of 3: loudness peaks later in the call (envelope rise fraction ↑, fall fraction ↓, envelope P1 ↑, P3 ↓, P4 ↑); amplitude-modulation depth ↑; share of calls with frequency jumps ↑; share of biphonation-like calls ↑ (rare, detector unvalidated).
- Confirmed 2 of 3: harmonic-to-fundamental ratio ↓ (non-monotonic: falls to about P12, then partly recovers); tonality (start, middle, end) ↓; envelope P2 ↓; f0 P6 ↓.
- Not confirmed: subharmonics; envelope P5, P7 and P8; f0 P4.
- Block gain over the baseline: E2L1 +0.18 [0.08, 0.27] (predicted +, **confirmed**); E2L2 −0.05 [−0.13, 0.04] (4 pups); E2L3 +0.25 [0.17, 0.34], 13/14 pups, but the locked prediction was "−" (inner check −0.02), so by the rule this is not confirmed. **1 of 3: not confirmed.** The beyond-contour block helped in two litters but the locked direction prediction was wrong for one of them. It is a candidate for a future pre-registered test on new data, not a finding.
- Envelope-mode block: 0 of 3 (E2L1 +0.14 [0.03, 0.24] against a "−" prediction). All selected features together: 0 of 3.

**Q3 Variability: the stereotypy hypothesis is contradicted.** Call-to-call variability **increases** with age:
- f0-mode score dispersion (P1, P2, P3) and log-duration dispersion: 3 of 3 litters;
- f0 median dispersion and within-family dispersion (f0 median, f0 P1): 2 of 3.

Duration dispersion is non-monotonic (it peaks around P11–15). Within P1–P21 there is no sign of calls becoming more stereotyped. Calls diversify within call types, not only through a changing type mix. The variability block adds no decoding information (0 of 3).

**Q4 What remains (exploratory, VAE seed 0).**
- The per-call residual age score (VAE embedding → age minus the classical + shape prediction) tracks the true residual on held-out litters with ρ = 0.39, 0.33 and 0.35.
- Occluding the fundamental moves the score most (mean |Δ| ≈ 0.6–0.7 days per third of the call).
- Occluding the **harmonic band** removes the most information: the ρ drop is 0.037 / 0.057 / 0.035 for the start, middle and end thirds of the harmonic, against ≤ 0.013 for each fundamental third.
- Harmonic effects are large at P1–7 and vanish by P15–21, when 2·f0 mostly lies above the 100 kHz analysis band.
- The latent traversal along the residual direction is visually uninformative: decoded images barely change, and the difference concentrates in a small fixed-position blob that looks like an artifact.

Reading: part of what the VAE adds beyond classical + shape is harmonic structure in young pups' calls. The band limit (Nyquist / fmax 100 kHz) makes this partly a function of f0 itself.

**Q5 Biological contrasts (exploratory).**
- Sex × age: no interaction survives FDR across the 24 confirmed features. The smallest p is 0.015 (harmonic ratio). 28 pups.
- Identity: pups are weakly but reliably identifiable on an unseen day within litter and age band. Balanced accuracy is 0.09–0.17 against shuffled 0.06–0.09 in the large litters, with permutation p < 0.05 in 5 of 6 large-litter cells.
- Signatures do not strengthen consistently with age: E2L1 rises from P1–7 to P8–14 and then plateaus; E2L3 is flat and then falls. Recording session is confounded with pup within a day.

## Status
- Developmental age trends in data-derived shape, envelope, modulation and variability features: **REPLICATED across held-out litters** (same cohort, locked predictions, FDR).
- Variability increases with age, contradicting stereotypy maturation within P1–21: **REPLICATED** (same standard).
- Added information beyond classical + H-M7a shape: **not shown** for any block. Beyond-contour is a candidate (positive CI in 2 of 3 litters, one against the locked direction).
- Harmonic band as the site of residual VAE information: **OBSERVED** (exploratory).
- Sex differences: **not supported**. Individual signatures: **OBSERVED**, weak and session-confounded.

## Limits
- One cohort, three litters. E2L2 has 4 pups (weak rotation).
- Rotations are not independent: each litter is in the discovery set twice.
- Most candidates trend with age, so "confirmed age trend" is a low bar. The block gain tests are the stringent ones.
- The nonlinear-phenomena detectors are not validated against hand labels.
- Harmonic visibility depends on f0 because of the 100 kHz analysis band.
- No mechanistic interpretation is implied.
