# Rat USV Acoustic Phenotyping

Reproducible rat ultrasonic-vocalization research, from immutable source evidence to cautious acoustic phenotyping and longitudinal representation research.

**Current status:** the analysis pipeline runs end to end on the frozen dataset (574 daily pup recordings, 28 pups, P1–P21): call detection, classical features and baselines, frozen animal/litter splits with negative controls, learned call representations, temporal organisation, explanation figures and a pre-registered discovery/confirmation round. Milestones M0–M5, M7 (first pass) and M9 are done; M6 (longitudinal state space) and M8 (external validation) are open. Thesis write-up is in progress. Aims: `docs/RESEARCH_CONTEXT.md`. Plan and per-milestone status: `docs/ROADMAP.md`. Decisions: `docs/decisions/`. Agent rules: `AGENTS.md`.

## Results so far

Evidence states follow `docs/RESEARCH_CONTEXT.md`; each row links to its decision record.

| Milestone | Result |
|---|---|
| M1 call detection | Hand-checked precision ≈ 0.93 (≈ 0.99 at peak SNR ≥ 14 dB), recall ≈ 0.99 ([0008](docs/decisions/0008-m1-call-detector.md)) |
| M2 classical baseline | Fundamental-contour features and mixed-model age trajectories; Avisoft count ρ = 0.98 ([0009](docs/decisions/0009-m2-classical-features-baseline.md)) |
| M3 splits + controls | Frozen animal and litter splits; a session leak found in leave-one-animal-out; classical features transfer across litters ([0010](docs/decisions/0010-m3-splits-negative-controls.md)) |
| M4 learned representations (H1) | VAE and CNN add age information beyond classical features, SSL does not; H1 INTERNALLY_VALIDATED ([0013](docs/decisions/0013-m4-results.md)). H2 observed but session-confounded, H2b not supported ([0014](docs/decisions/0014-h2-h2b-results.md)) |
| M5 temporal organisation | Call order adds nothing beyond single-call structure; H3 not supported ([0017](docs/decisions/0017-m5-temporal-organisation.md)) |
| M7 explanation | Call families, latent–property links, landscape figures, exploratory explorer. Contour shape absorbs ~76% of the VAE gain within cohort; independent test inconclusive ([0018](docs/decisions/0018-h-m7a-preregistration.md), [0019](docs/decisions/0019-h-m7a-results.md)) |
| M9 discovery/confirmation | Shape, envelope and modulation age trends and rising variability replicated in held-out litters; no feature block confirmed as adding age information beyond classical + shape ([0021](docs/decisions/0021-m9-results.md)) |

Negative and inconclusive results are kept on purpose. No mechanistic or causal claims are made.

## Start here

```bash
uv sync --frozen --group dev --group workflow
uv run --frozen usv --help
uv run --frozen pytest
uv run --frozen pre-commit install
```

Python 3.12 is selected by `.python-version`; uv.lock pins dependencies. The local machine already has an ignored `configs/data/local.yaml`. Elsewhere copy `configs/data/example.yaml` and set source/output paths. See REPRODUCIBILITY.md and SOURCE_LAYOUT.md.

## Analysis commands

`detect`, `detect-review`, `detect-score` (M1), `features`, `baseline` (M2), `controls` (M3), `call-images`, `m4`, `h2` (M4), `explain`, `explorer`, `landscape` (M7) and the M9 runners. Commands marked "needs `--group ml`" in `usv --help` require `uv sync --frozen --group ml`. Each run directory records the git commit, config, split, seed, metrics, predictions and figures.

## Data provenance commands (M0)

Every command writes a new directory under the configured output_root and prints its path. No command modifies a source artifact. Save each printed path to chain steps:

```bash
INV=$(uv run --frozen usv inventory --config configs/data/local.yaml)
AUD=$(uv run --frozen usv audit --config configs/data/local.yaml --inventory "$INV/source_inventory.parquet")
uv run --frozen usv crosswalk --config configs/data/local.yaml --inventory "$INV/source_inventory.parquet"
uv run --frozen usv dataset-atlas --config configs/data/local.yaml --inventory "$INV/source_inventory.parquet" --audit "$AUD/wav_audit.parquet"
uv run --frozen python scripts/initialize_registry.py
uv run --frozen usv validate --config configs/data/local.yaml --registry registry
# Explicitly re-read only files that have hashes in this baseline:
uv run --frozen usv validate --config configs/data/local.yaml --registry registry --source-inventory "$INV/source_inventory.parquet" --verify-hashes
```

`inventory`: file metadata, candidate roles, configurable bounded SHA-256 coverage, errors and exclusions. `audit`: WAV headers plus at most three sampled windows, channel-wise amplitude proxies and bounded hashes; not full-file decode certification. `crosswalk`: unresolved filename alias ledger only. `dataset-atlas`: technical JSON/Markdown report with biological counts explicitly unknown. `validate`: schema/identity checks plus optional source hash, split-lock and fit-scope checks. A valid empty registry is never M0-ready.

There is no automatic identity resolver, full Excel/R parser or MLflow integration. Hashing defaults are bounded; set `mode: full` deliberately for complete baseline coverage. Source symlinks are recorded/skipped, never followed. Inventory does not expand archives.

The bounded follow-on commands are documented separately: [`evidence-pilot` and `validate-evidence`](docs/M0_1_EVIDENCE_PILOT.md), [`full-decode-pilot`](docs/M0_2_FULL_DECODE_PILOT.md), [`review-packet`](docs/M0_3_REVIEW_PACKET.md), [`metadata-search`](docs/M0_4_METADATA_DISCOVERY.md), and [`reconcile-metadata`](docs/M0_5_METADATA_RECONCILIATION.md). They retain exact source locators, coverage gaps and candidate comparisons.

## Repository map

- `protocol/`: scientific contract, evidence states, exclusions and reviewer challenges.
- `registry/`: schemas, dictionary and ignored zero-row parquet placeholders.
- `configs/`: explicit acquisition/QC/preprocessing/model/experiment decisions.
- `src/rat_usv/`: provenance, QC, signal processing, features, representations, statistics, evaluation, explainability and visualization.
- `workflow/`: runnable inventory-only Snakemake skeleton; no training DAG.
- `tests/{unit,integration,scientific}/`: synthetic regression and scientific invariants.
- `docs/`: NASS roadmap, baselines, visualization, decisions and research log.
- `reports/`, `figures/`, `paper/`: templates and registries; review before publishing any derived content.
- `.local/`: private runs, setup report and environment caches; ignored.

Open follow-ups are in NEXT_STEPS.md. Scientific claims remain hypotheses until an appropriate validation milestone is met.
