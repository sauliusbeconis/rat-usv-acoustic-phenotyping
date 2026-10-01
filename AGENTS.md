# Agent guide

## Goal
Bachelor's thesis (Vilnius University, Department of Neurobiology and Biophysics; supervisor Osvaldas Ruksenas): results must be thesis-ready, i.e. clear methods, figures and honest limits.

Test whether the multidimensional structure of rat pup USVs (call morphology, timing, sequences) captures postnatal development (P1–P21) **beyond conventional summary variables**, as a reproducible trajectory through an acoustic state space that holds for unseen animals. Age prediction is a validation task, not the goal. Full brief: `docs/RESEARCH_CONTEXT.md` (read it once). Plan and current milestone: `docs/ROADMAP.md`.

## Data (done; do not re-audit)
The dataset is reconstructed and frozen. Load it with `rat_usv.data.linkage.load_recordings()`, or run `uv run --frozen usv linkage-check --config configs/data/local.yaml`. Details and caveats are in `private/data-linkage/README.md` (private). Animal name = marking `X.X.X.X`; key `animal_id` = `E2L3-0001`; age = days since the written birth date (P0). Do not reopen identity, age or dam questions unless new source evidence appears.

## Hard rules
1. Source audio and records are read-only. Write outputs only under `.local/runs/` or `private/`.
2. The animal is the unit. Split by animal (and test by litter where possible); never split randomly by call. Fit every learned step (scaling, feature selection, dimensionality reduction, SSL, tuning) on training animals only.
3. No test-set debugging. Freeze each split before use; keep failed runs and all seeds.
4. Private data, inventories and outputs never go into Git. Run `uv run --frozen python scripts/check_git_safety.py --staged` before any commit. Never push unless asked.
5. Claims follow `docs/RESEARCH_CONTEXT.md`: evidence states, negative controls, no mechanistic or causal language, and exploratory plots (UMAP etc.) are never proof.

## How to work
- Work milestone by milestone from `docs/ROADMAP.md`. Each milestone has a concrete output and a done-criterion; finish it, validate it, then update the roadmap status.
- Decide technical choices yourself (libraries, parameters, file formats, code structure) and record them in the run or a short decision note. Ask the owner only about genuine scientific trade-offs, batched into one question.
- Prefer simple, working code over infrastructure. Python 3.12, `uv`, typed code in `src/rat_usv/`, synthetic data only in tests. Notebooks are for exploration; logic that matters moves into `src/`.
- Every experiment run directory records the git commit, config, split, seed, metrics, predictions and figures (`rat_usv.provenance.io.new_run`).
- Check: `uv run --frozen ruff check . && uv run --frozen ruff format --check . && uv run --frozen mypy && uv run --frozen pytest`.
