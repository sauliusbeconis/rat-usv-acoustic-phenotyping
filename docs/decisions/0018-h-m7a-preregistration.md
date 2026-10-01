# 0018 — Pre-registration: test of H-M7a (contour shape)

Date: 2026-10-01. Written and committed **before** any shape feature was computed or any Fragile X model was fitted.

**H-M7a** (decision 0016, post hoc): the extra age information in learned call embeddings reflects contour shape, i.e. onset/offset frequency sweeps (U-shape) and fine frequency modulation (FM), which conventional summaries (median f0, slope, bandwidth) miss.

## Shape features (per call, from the M2 fundamental contour after octave correction and weak-frame trimming)
Let c(t) be the contour (kHz) over n frames, m = median(c), and e = max(1, round(0.1 n)) the edge length in frames.
1. `onset_sweep_khz` = mean(c[:e]) − m
2. `offset_sweep_khz` = mean(c[-e:]) − m
3. `u_index_khz` = (onset_sweep + offset_sweep) / 2 (positive = U-shaped)
4. `curvature_khz` = quadratic coefficient of a parabola fitted to c over normalised time τ ∈ [−1, 1] (positive = U)
5. `fm_residual_sd_khz` = SD of the residual after that parabola fit (fine FM)
6. `fm_extrema_per_10ms` = number of local extrema of c (3-frame median-smoothed, prominence ≥ 1 kHz) per 10 ms of call

Per pup-day: median and 90th percentile of each over confident calls (12 features); calls need ≥ 5 contour frames.

## Test A, this cohort (NOT independent: the hypothesis was formed on these held-out litters)
Protocol-v1 decoder and split (leave-one-litter-out, pup bootstrap, VAE seeds 0–2).
- A1. Directional prediction: per pup-day medians of `u_index_khz`, `fm_residual_sd_khz` and `fm_extrema_per_10ms` increase with age (mixed model, age-spline LRT plus a positive linear trend). Supported if ≥ 2 of 3 rise.
- A2. Added information: classical + shape beats classical (CI > 0).
- A3. Explanation: the VAE's gain over (classical + shape) is smaller than its gain over classical alone, averaged over seeds. Report the fraction absorbed.
Interpretation: at best "consistent with H-M7a"; not confirmatory.

## Test B, independent data (Fragile X rat pups, Zenodo 10.5281/zenodo.20940043, WT and KO separately)
Within their own DeepSqueak table (P3, P6, P10, P14, P21). A pup-day = one recording file with ≥ 5 accepted calls. Grouped 5-fold CV by `Rat`, same ridge pipeline.
- Baseline: per-file medians of Call Length, Principal Frequency, Delta Freq, Slope, Peak Freq, plus call count.
- Shape/FM: per-file median and p90 of Sinuosity and Frequency Standard Deviation, plus shares of modulated call labels (Trill, Inverted_U, Step_Up, Step_Down, Multi).
- B1 (primary, WT): baseline + shape/FM beats baseline in per-rat MAE (bootstrap over rats, CI > 0).
- B2: the same in KO (secondary).
- B3: direction: median Sinuosity and the modulated-label share increase with age in WT (Spearman over pup-days).
H-M7a's FM component counts as **supported on independent data** only if B1 holds. The U-shape component cannot be tested in Test B (no contour data in the table).

Known limits: some Fragile X rat IDs may repeat across litters; a rat that appears with multiple file prefixes is grouped by `Rat` as given. Ages are sparse (5 days), and some pup-days have few calls at P14/P21.
