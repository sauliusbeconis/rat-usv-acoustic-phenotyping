# 0003 — Bounded full-decode QC after the evidence pilot

Date: 2026-09-22. Status: bounded implementation; scientific review pending.

The request to continue M0 is implemented as a full-decode technical QC pilot on the same four WAVs already selected and exposed in M0.1. This is the separately bounded decode step described in the original M0 plan. No additional recordings are selected in this milestone. The exact private input evidence path and all source hashes are recorded by the run.

Validate RIFF lengths, chunk extents, data-frame alignment and classic PCM16 format, then decode sequentially as integer samples. Compare decoder header/frame counts with the independently parsed structure and verify EOF. Retain exact coverage, per-channel integer statistics, endpoint runs, long digital-zero runs, decoder errors and uninspected intervals. Do not infer noise, silence, detector performance, biological validity or exclusion from these diagnostics.

The committed reporting policy uses 262,144-frame blocks, retains zero runs lasting at least 0.01 seconds (ceiling to frames) and endpoint runs lasting at least one frame. These are explicitly technical reporting cutoffs, not scientific or exclusion thresholds. The bounded command accepts at most 12 previously exposed recordings and 1 GiB of selected files by default; this execution uses only the four M0.1 WAVs. It hashes sources before and after decoding and checks the earlier manifest hashes. Candidate-output overflow remains an explicit incomplete report. Sources and earlier derivatives are never overwritten.

Preserve the draft protocol and empty canonical registries. Log the additional full-waveform exposure. This milestone does not establish future test eligibility. Backup/freeze, annotation by two independent reviewers, identity/acquisition/reuse decisions, protocol v1, final splits and ML remain separate gates.
