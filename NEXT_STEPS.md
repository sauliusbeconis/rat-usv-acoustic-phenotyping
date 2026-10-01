# Next steps

**Current milestone: M3 — splits and negative-control harness** (see `docs/ROADMAP.md`).

Done: M1 (decision 0008) and M2 (decision 0009). Classical features and mixed-model trajectories are in `.local/runs/20261001T115113.379619Z-features-7b168652` and `.local/runs/20261001T120216.745814Z-baseline-0cbba073`.

1. Freeze animal-level splits: leave-one-animal-out (primary), plus leave-one-litter-out as a stress test. Store split files with SHA-256 digests.
2. Build the control harness on M2 features and `background_windows.parquet`. Each control uses simple, fixed decoders (regularized linear/logistic, nested grouped CV), not a model-design choice:
   - age from background-only audio (spectral summaries of call-free windows);
   - age from SNR/amplitude-only features;
   - calendar date vs age: the litters were born on different dates, so test whether date or age explains the features better;
   - animal-ID and litter decodability from features;
   - label permutation within the hierarchy.
3. Verify the DeepSqueak-equivalent feature definitions against the DeepSqueak source before M8.
4. **M4 starts with an explicit owner decision on model design** (AGENTS.md).
