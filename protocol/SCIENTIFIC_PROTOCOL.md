# Scientific protocol — v1 (locked 2026-10-01 before any M4 held-out comparison)

Locked by owner decisions on 2026-10-01 (decision 0011); its SHA-256 is recorded in `protocol/LOCK.sha256`. Any change needs a dated decision note stating what M4 results had been seen.

## Context
Bachelor's thesis, Vilnius University, Department of Neurobiology and Biophysics; supervisor Osvaldas Rukšėnas. The recordings were collected in the department's lab by a two-person team (the thesis author and B. Danilovaitė) and first analysed in B. Danilovaitė's bachelor's thesis, which is cited.

## Question
Can the multidimensional acoustic structure of rat pup USVs (spectrotemporal morphology, timing, sequence organization) capture postnatal development more completely than conventional summary variables? See `docs/RESEARCH_CONTEXT.md`.

## Hypotheses
- **H1 (primary):** USV structure contains information about postnatal developmental state beyond conventional acoustic summary variables.
- **H2:** development is a structured trajectory through a latent acoustic space, not a set of independent univariate changes.
- **H2b:** the acoustic state tracks physical maturation (body weight) beyond chronological age; tested within age with animal-level models.
- **H3:** temporal organization (intervals, bouts, transitions, entropy) adds information beyond single-call morphology.
- **H4 (planned, public external data):** learned representations transfer to an independent cohort, and a pharmacological or disease-model perturbation displaces animals in the same acoustic state space. Tested on public external datasets only (roadmap M8); no new experiment. If only call-feature tables are available, the test is limited to classical features.

## Design (reconstructed, M0)
A retrospective observational cohort: 28 Wistar pups from 3 litters and 2 dams that shared a cage. Each pup was recorded daily for 5 min in isolation on a heated pad, P1–P21 (P0 = written birth date), then weighed. 574 pup recordings. There is no experimental manipulation; age is the only designed factor. Sex and weight are measured covariates.

## Units and estimands
- The animal is the independent unit; calls, bouts and animal-days are nested repeated observations. Report animal N and observation N separately.
- **Primary split (owner decision):** leave-one-litter-out, scored per pup. Every littermate shares every daily session, so the owner's "session-blocked leave-one-animal-out" is identical to training on the other two litters. Leave-one-animal-out is reported only as a session-leak-inflated upper bound, beside the background-only control (decision 0010).
- **Primary estimand (H1):** the reduction in per-pup age MAE (days, at the animal-day level) when learned call embeddings are added to the M2 classical features, relative to classical features alone. Both use the same fixed decoder (median imputation + indicators, standardisation, ridge with inner grouped-CV alpha) under leave-one-litter-out. Encoders are fitted only on the training litters of each fold. Uncertainty comes from a pup-level bootstrap of paired MAE differences. H1 is supported only if the 95% CI excludes zero **and** the representation's background-window embeddings do not decode age across litters **and** the gain holds without `edit_suspected` recordings.
- **Representation ladder (owner decision):** (1) AVA-style VAE (Goffinet et al. 2021), the required learned baseline; (2) small supervised CNN; (3) contrastive self-supervised encoder. A later rung is credited only if it beats the earlier rungs and the classical features under the primary split. Call embeddings are pooled per animal-day (mean and spread).
- **H2 estimand:** consistency of held-out animals' trajectories in representation space (e.g. the age ordering of animal-day centroids, and the distance–age relationship), compared with univariate feature trajectories.

## Known confounds and limits
Within a litter, age is identical to recording session and cumulative handling. Calendar date and age are partly separable across litters. Litter and dam effects are only weakly estimable (3 and 2 units). 127 recordings may have been edited after acquisition; primary results are run with and without them. Chamber temperature is unreliable and excluded. Claims are limited to this cohort until an independent cohort is analysed.

## Evaluation rules
Only training animals are used for any learned step (normalization, selection, dimensionality reduction, SSL, tuning). Nested grouped cross-validation is used for tuning. Seeds, failures and compute budgets are recorded. The negative controls in `docs/ROADMAP.md` (M3) are mandatory before any H1/H2 claim. Exploratory projections are never evidence.
