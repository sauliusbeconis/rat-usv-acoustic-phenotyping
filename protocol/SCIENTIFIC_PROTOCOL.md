# Scientific protocol — draft v0.2, not locked

Lock this protocol (dated, with its digest recorded) before the first M4 held-out comparison is run. Changes after locking need a dated decision note.

## Context
Bachelor's thesis, Vilnius University, Department of Neurobiology and Biophysics; supervisor Osvaldas Ruksenas. The recordings were collected for an earlier bachelor's thesis in the same department and are reused with citation.

## Question
Can the multidimensional acoustic structure of rat pup USVs (spectrotemporal morphology, timing, sequence organization) capture postnatal development more completely than conventional summary variables? See `docs/RESEARCH_CONTEXT.md`.

## Hypotheses
- **H1 (primary):** USV structure contains information about postnatal developmental state beyond conventional acoustic summary variables.
- **H2:** development is a structured trajectory through a latent acoustic space, not a set of independent univariate changes.
- **H2b:** the acoustic state tracks physical maturation (body weight) beyond chronological age; tested within age with animal-level models.
- **H3:** temporal organization (intervals, bouts, transitions, entropy) adds information beyond single-call morphology.
- **H4 (future):** learned representations transfer to unseen animals, then to independent cohorts and perturbations.

## Design (reconstructed, M0)
A retrospective observational cohort: 28 Wistar pups from 3 litters and 2 dams that shared a cage. Each pup was recorded daily for 5 min in isolation on a heated pad, P1–P21 (P0 = written birth date), then weighed. 574 pup recordings. There is no experimental manipulation; age is the only designed factor. Sex and weight are measured covariates.

## Units and estimands
- The animal is the independent unit; calls, bouts and animal-days are nested repeated observations. Report animal N and observation N separately.
- **Primary estimand (H1):** in held-out animals (leave-one-animal-out), the improvement in recovering postnatal age at the animal-day level when the learned representation is added to classical features, compared with classical features alone. The learned comparator must include the AVA VAE baseline before any custom model is credited. Uncertainty comes from animal-level resampling. A difference counts only if its CI excludes zero and every M3 negative control stays at chance.
- **H2 estimand:** consistency of held-out animals' trajectories in representation space (e.g. the age ordering of animal-day centroids, and the distance–age relationship), compared with univariate feature trajectories.

## Known confounds and limits
Within a litter, age is identical to recording session and cumulative handling. Calendar date and age are partly separable across litters. Litter and dam effects are only weakly estimable (3 and 2 units). 127 recordings may have been edited after acquisition; primary results are run with and without them. Chamber temperature is unreliable and excluded. Claims are limited to this cohort until an independent cohort is analysed.

## Evaluation rules
Only training animals are used for any learned step (normalization, selection, dimensionality reduction, SSL, tuning). Nested grouped cross-validation is used for tuning. Seeds, failures and compute budgets are recorded. The negative controls in `docs/ROADMAP.md` (M3) are mandatory before any H1/H2 claim. Exploratory projections are never evidence.
