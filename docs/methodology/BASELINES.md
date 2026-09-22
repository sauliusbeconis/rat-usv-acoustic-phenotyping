# Baseline design before advanced models

Implement only after M0 confirms acquisition, identities and endpoint. Reconstruct historical features from their definitions and units first; do not assume spreadsheet columns have identical meaning across exports. Keep disagreements with historical results as a provenance report.

| Level | Proposed analysis | Reason / required evidence |
|---|---|---|
| Descriptive | Per-animal/day count, duration/frequency/bandwidth distributions, intervals, missingness and weights | Establish what data measure and expose confounds; report animal/litter N separately |
| Classical | Endpoint-appropriate hierarchical regression / smooth age effect | Account for animal repeats and litter; check assumptions and uncertainty, avoid unjustified random-effect complexity |
| Linear | Regularized linear/logistic model, defined by endpoint | Transparent incremental information over conventional features |
| Tree | Modest tree ensemble with grouped tuning | Nonlinear comparator on the same features/partitions |
| Spectral | Small CNN | Minimal learned acoustic comparator |
| Representation | Frozen SSL encoder + linear probe, then constrained fine-tuning | Separate representation quality from downstream capacity |
| Hierarchical | Pooled, ordered sequence, state-space | Incremental temporal information, paired ablations |

Negative controls: background/no-USV windows; acquisition date/session alone; amplitude/noise alone; litter prediction; duplicate hashes; hierarchy-respecting shuffled labels; random encoder; count-matched subsampling; temporal-order shuffling. Their feasibility depends on M0 metadata. A positive nuisance control triggers a design/claim review, not an automatic technical fix.

Each experiment must declare population, estimand, endpoint, independent units, allowed fit data, split hash, metrics, uncertainty, tuning/compute budget, seeds and failure criteria. No call-level accuracy may substitute for the animal-level primary endpoint. Report per-animal aggregation and uncertainty, not only pooled performance. Detector accuracy and contour reliability require separate manual annotation validation before using those measurements as biological evidence.
