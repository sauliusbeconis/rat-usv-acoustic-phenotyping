# Next steps

**Current milestone: M1 — call detection, finishing** (see `docs/ROADMAP.md`, decision 0008).

Done: detector (`usv detect`), full run over 574 recordings (42,391 calls, background windows, Avisoft agreement, calls per animal-day), and an external sanity benchmark.

1. **Owner:** open the latest `.local/runs/*-detect-review-*/review.html` in a browser, label all 225 items (call / not a call / unsure), click *Download labels*, then run
   `uv run --frozen usv detect-score --config configs/data/local.yaml --review-run <that run> --labels <labels.json>`.
2. If the share of real calls among detector-only events is low (e.g. under 0.5), tighten the detector (tonality/SNR), re-run `usv detect` and the review on new events. Otherwise freeze the detect run as the M1 output.
3. Mark M1 done in the roadmap and start M2 (classical per-call features including DeepSqueak-equivalent definitions; per animal-day summaries; mixed-model age trajectories).
