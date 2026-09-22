# 0005 — Git delivery of validated M0 infrastructure

Date: 2026-09-22. Status: engineering delivery; scientific review pending.

The continuation request explicitly permits pushing completed work to Git. Deliver the existing M0 through M0.3 implementation, generic methodology and usage documentation, dependency lock, synthetic tests and CI. No scientific gate is advanced by this delivery.

Review every candidate path and staged content before committing. Remove machine-specific source paths, inventory counts and source-layout observations from Git documentation, preserving the original local notes in an ignored delivery audit. Keep private inputs, manifests, reports, registries and local agent state out of the index. Extend the data guard to reject `.agents` and `.codex` directories even when added explicitly.

Acceptance requires the five AGENTS.md checks, candidate-checkout validation, a synthetic workflow smoke run and inspection of the populated staged diff. Push only a normal fast-forward to the configured repository, then verify the remote commit and CI result. Preserve failures and distinguish local validation from remote CI status.

Original marking, birth and session evidence, acquisition interpretation, allocation, access/reuse rights and named scientific review remain unresolved. Protocol v1, final splits and model fitting remain later gates.
