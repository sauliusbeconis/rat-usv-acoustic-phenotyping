# Rat USV Acoustic Phenotyping

Reproducible rat ultrasonic-vocalization research, from immutable source evidence to cautious acoustic phenotyping and longitudinal representation research.

**Current status:** the dataset is reconstructed and frozen (M0 done): 574 daily pup recordings, 28 pups, P1–P21. Next milestone: call detection (M1). Aims: `docs/RESEARCH_CONTEXT.md`. Plan: `docs/ROADMAP.md`. Agent rules: `AGENTS.md`. No models are trained yet and no novelty is claimed; NASS is a working concept.

## Start here

```bash
uv sync --frozen --group dev --group workflow
uv run --frozen usv --help
uv run --frozen pytest
uv run --frozen pre-commit install
```

Python 3.12 is selected by `.python-version`; uv.lock pins dependencies. The local machine already has an ignored `configs/data/local.yaml`. Elsewhere copy `configs/data/example.yaml` and set source/output paths. See REPRODUCIBILITY.md and SOURCE_LAYOUT.md.

## Implemented CLI

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

There is no training, automatic identity resolver, full Excel/R parser or MLflow integration yet. Hashing defaults are bounded; set `mode: full` deliberately for complete baseline coverage. Source symlinks are recorded/skipped, never followed. Inventory does not expand archives.

The bounded follow-on commands are documented separately: [`evidence-pilot` and `validate-evidence`](docs/M0_1_EVIDENCE_PILOT.md), [`full-decode-pilot`](docs/M0_2_FULL_DECODE_PILOT.md), [`review-packet`](docs/M0_3_REVIEW_PACKET.md), [`metadata-search`](docs/M0_4_METADATA_DISCOVERY.md), and [`reconcile-metadata`](docs/M0_5_METADATA_RECONCILIATION.md). They retain exact source locators, coverage gaps and candidate comparisons. Scientific decisions remain unresolved.

## Repository map

- `protocol/`: scientific contract, evidence states, exclusions and reviewer challenges.
- `registry/`: schemas, dictionary and ignored zero-row parquet placeholders.
- `configs/`: explicit acquisition/QC/preprocessing/model/experiment decisions.
- `src/rat_usv/`: implemented inspection, provenance, validation and atlas code; named namespaces for later work.
- `workflow/`: runnable inventory-only Snakemake skeleton; no training DAG.
- `tests/{unit,integration,scientific}/`: synthetic regression and scientific invariants.
- `docs/`: NASS roadmap, baselines, visualization, decisions and research log.
- `reports/`, `figures/`, `paper/`: templates and registries; review before publishing any derived content.
- `.local/`: private runs, setup report and environment caches; ignored.

Follow NEXT_STEPS.md for a read-only Plan-mode M0 audit. Scientific claims remain hypotheses until an appropriate validation milestone is met.
