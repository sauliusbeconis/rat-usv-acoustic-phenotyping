# 0013 — M4 results: learned call representations vs classical features (H1)

Date: 2026-10-01. Protocol v1 (locked, `protocol/LOCK.sha256`), settings in decision 0012. Primary split: leave-one-litter-out, scored per pup (28 pups, 3 folds). All 9 runs (3 rungs × seeds 0–2) are reported; the table is in `.local/runs/m4_overnight_table.json` (log `overnight_m4_20261001T130401Z.log`, all exits 0).

## Primary estimand: per-pup age MAE, classical − (classical + learned), pup bootstrap 95% CI
| Rung | Seed 0 | Seed 1 | Seed 2 | Background-embedding ρ (criterion < 0.2) | Without edit-suspected |
|---|---|---|---|---|---|
| VAE | +0.74 [0.59, 0.88] | +0.66 [0.50, 0.81] | +0.59 [0.44, 0.73] | 0.08 / 0.07 / 0.04 ✓ | holds ✓ |
| CNN (supervised on age) | +0.99 [0.82, 1.16] | +1.06 [0.90, 1.23] | +1.08 [0.90, 1.25] | 0.12 / 0.12 / 0.17 ✓ | holds ✓ |
| Contrastive SSL | +0.53 [0.39, 0.67] | +0.82 [0.64, 1.01] | +0.76 [0.61, 0.92] | 0.21 / 0.16 / 0.24 ✗ in 2 of 3 | holds ✓ |

Classical-only per-pup MAE is 3.42 days. Seed-averaged combined MAE: VAE 2.76, CNN 2.38, SSL 2.72. CNN embeddings alone reach 2.40 (vs 3.42; +1.03 [0.86, 1.20], 27/28 pups). The gain appears in every held-out litter (combined vs classical: VAE 2.78/2.45/2.84, CNN 2.60/2.18/2.27, SSL 2.77/2.44/2.76 for E2L1/E2L2/E2L3, vs 3.34/3.31/3.51).

## Ladder (credit only if better than earlier rungs and classical)
- **VAE: credited.** Beats classical: +0.66 [0.53, 0.79], 27/28 pups (seed-averaged).
- **CNN: credited.** Beats VAE: +0.38 [0.24, 0.52], 25/28 pups.
- **SSL: not credited.** Worse than CNN (−0.34 [−0.48, −0.21]), no better than VAE (+0.04 [−0.04, 0.13]), and its background-window embeddings decode age slightly across litters in 2 of 3 seeds (ρ 0.21, 0.24). The augmentations may leave background or session texture in the embedding; recorded as a failed rung.

## Secondary analysis (not the locked estimand): richer classical baseline
Per animal-day 10/50/90th percentiles of 12 call features plus jump and short-call shares (61 features) improve the classical decoder from 3.42 to 3.01 days. Learned embeddings still add information in all 9 runs: VAE +0.32 to +0.43, CNN +0.59 to +0.66, SSL +0.35 to +0.52 days (all CIs > 0). `.local/runs/m4_rich_baseline_secondary.json`.

## Claim status
**H1 → INTERNALLY_VALIDATED** within this cohort: multidimensional call structure (learned embeddings) carries postnatal age information beyond conventional and distributional classical summaries. The result holds in pups from held-out litters, across seeds, without possibly edited recordings, and is not reproduced by background-window embeddings (VAE, CNN).

Not established: generalisation to other cohorts or labs (M8); separation of development from cumulative handling and session (inseparable within litter); *which* acoustic structures carry the extra information (M7 explanation chain); H2 trajectory structure. Limitations: 3 litters, 2 dams in one cage, a single lab.

Smoke-run disclosure: decision 0012.
