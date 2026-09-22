# Exclusions and unresolved observations

No source has been biologically excluded by the scaffold. A parser failure, hash-budget skip or symlink skip is a tooling status, not an exclusion.

Every future exclusion requires exclusion_id, source/recording/animal scope, original status, proposed reason, observable criterion, evidence locator/hash, author/date, whether outcomes were inspected, review status, affected analyses and sensitivity analysis. Never delete its registry row or original file. Duplicate files remain inventoried; a reviewed analysis-selection decision determines which observation contributes once.

Proposed review categories (thresholds not yet fixed): unreadable/truncated audio, acquisition incompatibility, off-protocol session, uncertain identity, duplicate observation, absent required metadata. Missing weight/sex alone does not justify deleting usable audio from unrelated endpoints. Report denominators before and after each decision, by animal/litter/day where established.

Ledger header: `exclusion_id,entity_type,entity_id,reason,evidence_ids,proposed_utc,proposer,outcomes_seen,status,reviewer,affected_claims,sensitivity_run_ids`.
Private populated ledger lives under .local until publication review.
