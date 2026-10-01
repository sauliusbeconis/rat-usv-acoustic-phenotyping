# Next steps

**Current milestone: M1 — call detection** (see `docs/ROADMAP.md`).

1. Run `uv run --frozen usv linkage-check --config configs/data/local.yaml` to confirm that the frozen dataset and all source WAVs are reachable.
2. Choose and implement a USV detector in `src/rat_usv/signal/`. A transparent spectrogram-energy detector is a reasonable first choice; a learned detector is fine if validated. Keep the band below the ~107 kHz Nyquist limit.
3. Build a stratified validation sample (litters × early/mid/late age × edit-suspected or not). Prepare hand-checking material for the owner (spectrogram pages with proposed calls), then measure precision and recall.
4. Run detection over all 574 final recordings plus background-only windows; write `calls.parquet` to a new `.local/runs/` directory.
5. Update the roadmap status.

History: the M0 audit tooling (`usv inventory`, `evidence-pilot`, … `reconcile-metadata`) and its docs (`docs/M0_*.md`, `docs/decisions/`) reconstructed the dataset. They are kept for provenance and are not needed for new work.
