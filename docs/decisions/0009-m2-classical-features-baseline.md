# 0009 — M2 classical features and baseline trajectories

Date: 2026-10-01. Status: implemented; results are **OBSERVED / exploratory** until the M3 negative controls run.

## Per-call features (`src/rat_usv/signal/features.py`, `usv features`)
Computed on the detector's whitened spectrogram (same noise floor as detection), per detected call:
- **Fundamental (f0) contour.** Per frame, the loudest tonal band. If a sub-peak at half its frequency is ≥ 25 kHz and within 15 dB, that sub-peak is taken as f0, because the loudest band can be the 2nd harmonic. Frames sitting at about 2× (or 0.5×) the call's median f0 are octave-corrected. Frames more than 10 dB below the call's median power (weak tails, echo fragments) are dropped. A 3-frame median filter is applied.
- **Features:** duration; f0 start, end, min, max, mean, median and SD; bandwidth (max − min f0); slope (linear fit, kHz/s); sinuosity (contour arc length / chord in ms × kHz units); frequency jumps (steps ≥ 10 kHz that persist ≥ 3 frames on both sides); harmonic share; tonality (1 − normalised spectral entropy); peak frequency; peak and mean SNR.
- **Harmonics** are counted only where 2·f0 lies inside the analysed band (≤ 100 kHz; the Nyquist frequency is 107 kHz). Above f0 ≈ 49 kHz the harmonic cannot be observed, so `harmonic_observable_share` is stored. Animal-day harmonic medians use only calls observable for at least half their frames, with ≥ 5 such calls.
- **Confident call set** = peak SNR ≥ 14 dB (decision 0008). Animal-day morphology medians need ≥ 5 confident calls.
- QC: fundamental tracking was checked visually on randomly chosen and edge-case calls. The remaining "jumps" are mostly real two-note calls or low-SNR events. Very short (2–5 ms) high-frequency (65–72 kHz) tonal blips are kept as a real short-call type.
- *DeepSqueak equivalence* (needed for M8): slope, sinuosity, tonality and frequency SD are defined analogously but **not yet verified** against DeepSqueak's code. Verify before any comparison with the Fragile X table.

## Baseline model (`src/rat_usv/statistics/trajectory.py`, `usv baseline`)
For each animal-day outcome: a linear mixed model `y ~ cr(age, df=4, centred) + sex + litter` with a random intercept per animal (statsmodels MixedLM). The age effect is tested by an ML likelihood-ratio test against the same model without age; curves come from the REML fit (fixed effects averaged over the observed sex and litter mix, 95% CI from fixed-effect covariance). Call rate, duration and inter-call interval are modelled on the log scale (call rate + 0.1). Variants: primary; without `edit_suspected` recordings; call rate from all detections.

## Results (features run `20261001T115113.379619Z-features-7b168652`, baseline run `20261001T120216.745814Z-baseline-0cbba073`)
- 42,391 calls, of which 38,170 are confident. 494 of 574 animal-days have ≥ 5 confident calls; 422 have ≥ 5 harmonic-observable calls.
- **Sanity check against the 2020 Avisoft analysis** (571 recordings): call count per recording Spearman ρ = 0.98; median frequency per recording ρ = 0.84, median difference +0.19 kHz.
- **Every outcome depends strongly on age** (28 animals; age LRT p from 2.5×10⁻⁶ to 5×10⁻⁷⁹). Population shapes:
  - Call rate: about 10/min at P1, a dip around P5–P7 (about 3/min), a peak around P13 (about 15/min), then a decline to about 1/min at P21.
  - f0: flat at about 44 kHz to P10, rising to about 58 kHz by P16–P17.
  - Duration: about 55 ms falling to about 20 ms.
  - Harmonic share (observable calls only): about 0.87 falling to about 0.1 by P16.
  - Tonality: highest at mid-ages and falling after P15.
  - Inter-call interval: rises late.
- Between-animal variance is small relative to day-to-day variance (animal ICC 0–0.16).
- **Robustness:** the same conclusions hold without the 127 `edit_suspected` recordings, and the call-rate trend holds with all detections.

## Interpretation limits
These are **OBSERVED** descriptive trajectories. Within a litter, age is identical to recording session and cumulative handling. Litter is a fixed effect with only 3 levels. The heated chamber removes the cold stimulus, which may explain the early-age call-rate dip, but this is a hypothesis. Nothing here addresses H1 (information beyond classical features); that is M4. M3 negative controls (date versus age, background-only, SNR/amplitude) must run before these trajectories are presented as developmental evidence.
