# Next steps

**Current milestone: M2 — classical phenotype + baseline** (see `docs/ROADMAP.md`).

M1 is done (decision 0008): detect run `.local/runs/20261001T103654.209131Z-detect-55e24631` (`calls.parquet`, `background_windows.parquet`, `recordings_detection.parquet`), hand-checked by the owner.

1. Per-call features from the audio of each detected call: fundamental-frequency contour (track f0, not the louder harmonic), duration, start/end/min/max/mean f0, bandwidth, slope, curvature/sinuosity, tonality, amplitude/SNR, harmonic presence, frequency-jump count; plus DeepSqueak-equivalent definitions for the M8 Fragile X comparison.
2. Per animal-day summaries: call count and rate, inter-call intervals, bouts.
3. Mixed-model age trajectories (animal random effect; sex; litter as a fixed effect or sensitivity) and individual-animal trajectory figures with CIs (evidence figures).
4. Use `peak_snr_db >= 14` as the primary call set and all detections as a sensitivity analysis.
