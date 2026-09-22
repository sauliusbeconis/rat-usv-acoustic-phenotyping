# Reproducibility contract

## Environment

Use `uv sync --frozen --group dev --group workflow`. `.python-version` selects Python 3.12 and uv.lock resolves package versions. No global Python, CUDA, RStudio or GPU assumption is needed for M0. Source audit uses python-soundfile/libsndfile read mode; record the runtime decoder version for acquisition review.

On a constrained sandbox use local caches:

```bash
export UV_CACHE_DIR="$PWD/.local/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.local/python"
export XDG_CACHE_HOME="$PWD/.local/cache"
export PRE_COMMIT_HOME="$PWD/.local/cache/pre-commit"
uv sync --frozen --group dev --group workflow
```

## Evidence chain

Each CLI run writes immutable-by-convention new files in a unique timestamp/UUID directory. `run.json` records command, UTC time, source/output configuration, Python/platform/package versions, Git commit/dirty status, content digest of src/config, uv.lock hash, input manifest hashes, libsndfile version and output hashes. `code_snapshot.json` preserves the actual src/config text (excluding local.yaml, whose resolved values are recorded in run.json), including uncommitted code. Initial bootstrap runs before this refinement lack that snapshot; the final setup run index points to runs with it. Runs are not cryptographically signed and must be backed up separately. Failed runs may leave partial directories; absence of run.json means incomplete. Do not reuse them as completed evidence.

Inventory SHA-256 covers exact file bytes, not decoded acoustic equivalence. WAV copies with different headers need a later, separately specified PCM-content duplicate audit. File-level hashing is chunked; `none`, `budget` and explicit `full` modes are supported. Per-file and total byte limits apply in budget mode; summary reports actual bytes and statuses. Stat-before/after detects common concurrent changes; use a stable source snapshot for a scientific freeze. `checksums.json` uses JSON-quoted relative paths; it is not sha256sum syntax. `registry/checksums.sha256` is an empty local placeholder until a full, reviewed freeze.

A limited initial hash baseline cannot establish immutability of unhashed files. `validate --verify-hashes` re-reads the supplied hashed subset and reports coverage. Initial setup also compares original path/size/mtime_ns snapshots. It does not claim cryptographic verification of the entire source tree.

## Reruns and reports

Use paths printed by CLI commands; never guess a latest directory during a concurrent run. Config relative paths resolve from the config file, not shell cwd. Registry outputs contain schemas but no fabricated identities. Full source inventory and actual M0 tables remain ignored/private; exported release manifests require deliberate de-identification and access review.

Test with `uv run --frozen pytest`; CI needs no private tree, GPU or credentials. Snakemake supports `uv run --frozen --group workflow snakemake -s workflow/Snakefile --cores 1 --dry-run`; the current DAG only creates an inventory run and completion marker. Remove its generated `.local/workflow/inventory.done` marker to request a new snapshot. Do not mistake that marker for source freshness monitoring.

Future experiment records must include protocol version, data/split hashes, complete preprocessing fit IDs, seeds, optimizer/stopping criteria, dependencies, hardware, failure logs and every attempted hyperparameter/seed. Statistical resampling follows animal/litter structure; quantify model and data uncertainty separately. External validation requires a locked pipeline and a cohort appropriate to the claim.

References: [uv project workflow](https://docs.astral.sh/uv/guides/projects/), [soundfile read/seek API](https://python-soundfile.readthedocs.io/).
