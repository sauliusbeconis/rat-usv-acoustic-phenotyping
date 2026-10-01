# Next steps

## Tomorrow (2026-10-02): M9, data-driven discovery (plan: decision 0020)
Start with work package 0, then go in order: 1 → 2 → 3 → 4 (runner) → 5 → 6 → 7. About one working day.

Inputs (all private, `.local/runs/`):
- detect run `20261001T103654.209131Z-detect-55e24631`
- features run with shape `20261001T164535.599872Z-features-c17ba600`
- controls + frozen splits `20261001T122013.275165Z-controls-c4cf4161`
- call images `20261001T124005.643225Z-call-images-c60840b7`
- M4 table `m4_overnight_table.json` (VAE seed 0–2 runs); explain run `20261001T153401.888949Z-explain-1a8315ae`

Rules: discovery code must never load the held-out (confirmation) litter; lock and hash predictions before confirmation; FDR q = 0.05; confirmed = ≥ 2 of 3 rotations; baseline = classical + six H-M7a shape features.

## Status so far
| Hypothesis | Status |
|---|---|
| H1 learned call structure adds age information beyond classical features | INTERNALLY_VALIDATED (0013) |
| H-M7a contour shape explains most of it | Consistent within cohort, ~76% of the VAE gain; external test inconclusive (0018, 0019) |
| H2 trajectory through acoustic space | OBSERVED, session-confounded (0014) |
| H2b acoustic state tracks weight | Not supported (0014) |
| H3 call timing or order adds information | Not supported (0017) |

## Later
- External replication: ask the Fragile X authors (Auerbach lab) for audio (owner and supervisor).
- Thesis support: figure registry, methods from decisions 0008–0020, results tables.
