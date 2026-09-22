# Contributing

Start with AGENTS.md. Propose one bounded milestone and identify the hypothesis or validity risk it addresses. Describe input/output schemas, unknowns, provenance and acceptance checks before changing analysis.

Use `uv sync --frozen --group dev --group workflow`, then `uv run --frozen pre-commit install`. Run ruff check/format, mypy and pytest before review. Synthetic fixtures must be generated in temporary directories; never copy private samples into tests. Tests should fail on leakage, provenance loss or meaningful behavior regressions.

Record substantive methodology decisions in docs/decisions/. Preserve failed experiments in reports/experiment_cards/ with status FAILED and a private run reference; redact source identifiers before committing public cards. Changes after protocol or split lock require a dated amendment and a separately labelled exploratory analysis if test information has been seen.

Before committing: inspect `git status --short`, `git diff --cached --stat`, `git diff --cached`, and run `uv run --frozen python scripts/check_git_safety.py --staged`. Ignore rules are not a confidentiality guarantee: the guard blocks common artifacts but cannot recognize private values in arbitrary text. Review all tracked contents. Never use `git add -f` on data. Do not push without an explicit request.

Do not change uv.lock manually. Use uv to update a named dependency and explain why; rerun affected validation. Cite software releases used in the paper. Dataset and software licensing, authorship and release permissions require owner decisions; no dataset license is inferred from repository access.
