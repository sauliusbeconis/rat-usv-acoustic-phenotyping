# Scientific and engineering SOP

Priority order: **scientific validity > provenance > reproducibility > leakage prevention > biological interpretation > uncertainty > explainability > model performance**.

## Current scope
M0 infrastructure is implemented. Actual identity reconstruction, protocol lock, final splits and ML are NOT complete. Read NEXT_STEPS.md and protocol/SCIENTIFIC_PROTOCOL.md first. Inspect, plan, implement a bounded milestone, then validate. Do not reinterpret scaffold completion as approval for training.

## Non-negotiable rules
- Raw/source artifacts are immutable. Never rename, move, rewrite, normalize or delete them. Open read-only. Write new derivatives only to an explicit output location excluded from source traversal.
- The animal, not the call, is the biological unit. Calls are nested in bouts, sessions and animal-days; retain animal/litter/dam hierarchy. Litter may be the independent unit for litter-level exposures; document the estimand and experimental allocation.
- No random call split for primary claims. Group by animal; assess litter holdout and technical batches. Fit scalers, imputers, feature selectors, dimensionality reduction and SSL only on permitted training data for primary inductive claims. Any transductive analysis needs a separate protocol and claim.
- No test-set debugging or hyperparameter selection. Lock split bytes and separately reviewed digest before use. Keep fit-ID provenance. A lock file is an audit control, not tamper-proof storage.
- No silent exclusions, duplicate removal, identity merging, conflict resolution or missing-value filling. Preserve all aliases, conflicting assertions, units, locators and original values. Null means unknown, never zero.
- Do not execute historical R/Rmd or deserialize historical RData/RDS automatically. Inspect code and data dependencies first in a controlled environment; inspect spreadsheet formulas and cached values separately.
- UMAP/t-SNE/3D plots are exploratory, never proof. No causal overclaiming from observational prediction, latent factors, attention or attribution maps.
- Preserve failed experiments with configuration, logs, errors and interpretation. Never selectively report only favorable seeds/runs.
- Every scientific claim must link to a claim ID, versioned protocol, data/checksum manifest, run, code/config/split hashes and registered figure. No fabricated biological values, performance, uncertainty, citations or novelty claims.
- NASS is a working internal concept; no established novelty or validated neurobiological meaning.
- Keep all raw/private materials, private inventories and reports out of Git. No remote push unless explicitly requested. Before any commit inspect git status, staged diff and run the data guard.

## Engineering
Use Python 3.12+, pathlib, uv.lock and typed src code. Synthetic fixtures only in CI. Notebooks are exploratory; source of truth is src/config/workflow. New CLI commands must do real work before advertising them. Avoid hidden cwd assumptions; config paths resolve from the YAML file. Preserve leading zeros in aliases.

Validation: `uv run --frozen ruff check .`, `uv run --frozen ruff format --check .`, `uv run --frozen mypy`, `uv run --frozen pytest`, `uv run --frozen python scripts/check_git_safety.py --staged`.

Stop the affected scientific step when identities, acquisition semantics, consent/access rights, allocation, primary endpoint or split validity lack evidence. Continue independent infrastructure work. Record the specific unresolved decision rather than silently guessing.
