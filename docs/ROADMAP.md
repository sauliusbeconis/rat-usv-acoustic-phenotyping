# Roadmap

This work is a **bachelor's thesis** (Vilnius University, Department of Neurobiology and Biophysics; supervisor Osvaldas Rukšėnas). M1–M4 form the thesis core; M5 and M7 strengthen it; M6 (NASS) is optional and only justified by an M4/M5 gap. M8 (external / perturbation cohort) is planned and runs in parallel; its data must be obtained first.

Each milestone has one main output and a done-criterion. Work them in order; update the **Status** column when a milestone is done (date + run path). The aims are in `docs/RESEARCH_CONTEXT.md`. Methods are the agent's call unless marked *owner decision*.

| # | Milestone | Main output | Done when | Status |
|---|---|---|---|---|
| M0 | Data linkage | `private/data-linkage/recordings.csv` (+ animals, litters/dams) | Every WAV mapped to animal, date, age, sex, litter, dam, weight | **Done 2026-10-01** |
| M1 | Call detection | `calls.parquet`: one row per detected call in all 574 final recordings (start, end, frequency bounds, SNR), plus matched background-only windows | Detector precision and recall measured on a stratified hand-checked sample, reported per file and per age band; **calls per animal-day across P1–P21 reported** (if late days are near-empty, set the morphology window from the data and analyse call rate separately). Candidate detectors: ContourUSV, USVSEG, DeepSqueak or a transparent energy detector, chosen on our sample | Next |
| M2 | Classical phenotype + baseline | Per-call features (duration, peak/min/max frequency, bandwidth, contour slope and shape, amplitude) and per animal-day summaries (count, rate, inter-call intervals, bouts); mixed-model age trajectories | Individual-animal trajectories P1–P21 with CIs (evidence figures); thesis summaries reproduced as a sanity check | |
| M3 | Splits + negative-control harness | Frozen animal-level splits (leave-one-animal-out, leave-one-litter-out stress test); control suite run on M2 features | Controls reported: background-only, amplitude/SNR-only, animal-ID and litter decodability, recording-date vs age, label permutation | |
| M4 | Beyond classical (primary hypothesis) | Learned call representations: **AVA VAE (Goffinet et al. 2021) is the required learned baseline**, then a small spectrogram CNN and SSL trained on training animals only, all compared with M2 features under M3 splits | Animal-level incremental-information test with CIs; controls re-run on the representation; trajectory consistency across held-out animals | |
| M5 | Temporal organization (H2) | Bout/sequence features and a sequence model vs pooled call representations | Order-shuffle and bout-boundary ablations reported | |
| M6 | Longitudinal state space (NASS) | `z_animal(t)` model over P1–P21 | Beats the M4/M5 baselines on held-out animals, or the null is recorded. Start only if M4/M5 leave a reproducible gap | |
| M7 | Explanation + visualization | Latent → call families → acoustic properties → real calls; interactive explorer (point → animal, day, spectrogram, contour, audio); acoustic-landscape figures | Every concept figure is linked to a quantitative result and labelled by visualization class | |
| M8 | External / perturbation validation (H4) | An independent or drug-treated rat pup cohort processed by the frozen M1–M4 pipeline, with no refitting on its animals | The M4 representation and trajectory are tested on the new cohort; perturbation displacement compared with classical measures; null results recorded | Data to be obtained |

An exploratory explorer (M7 style) may be built right after M1 to help look at the data. It is labelled exploratory.

## Design facts that bound every claim
- **Cohort:** one lab and one cohort: 28 pups, 3 litters, 2 dams (both dams in one cage; one dam had two litters). Daily recordings P1–P21, 5 min each, Avisoft at 214285 Hz (usable band ≤ ~107 kHz). One litter has 4 pups; one lacks P17.
- **Splits:** primary evaluation is leave-one-animal-out. Litter holdout is a stress test only (3 litters). Dam-level generalization is not testable.
- **Age vs session:** within a litter, age is identical to recording day and session. Litters were born on different dates, so calendar date and age are partly separable across litters; use that in the M3 date control. Cumulative handling also grows with age; name it as a confound.
- **Possibly edited audio:** 127 recordings were likely trimmed or edited in 2020 (`edit_suspected`). Run every primary result with and without them.
- **Chamber temperature** was noted unreliably and is excluded.
- **Compute:** RTX 4070 (12 GB); about 74 GB of source audio. Cache decoded or derived arrays under `.local/`, never next to the sources.

## Positioning (literature check 2026-10-01)
Already established: USVs change with age (rat isolation calls peak around P7–11; frequency falls with growth); learned VAE spaces beat hand-picked features (Goffinet et al. 2021, eLife 10:e67855); mouse pup ages separate in VAE space (PMC9805956); rat pup VAE clusters and syntax in a disease model at P6/9/12 (Sci Rep 2025, PMC12304202). Our distinct contribution: **daily P1–P21 sampling of identified individuals with weights**, a held-out-animal, control-backed test of information beyond strong classical features, and whether acoustic state tracks physical maturation (weight) beyond chronological age. Main risk: few calls on late days (heated chamber, natural decline after the second week).

## Owner decisions still open
- **Obtain an external or drug-treated cohort for M8** (needed for EXTERNALLY_VALIDATED claims and H4). Options: a new recording in the department's lab under the same protocol (Avisoft, 5-min isolation, P1–P21) with a pharmacological arm; or existing data from published rat pup studies (e.g. the 2025 Sci Rep model study, PMC12304202), requested from the authors. Agree the design or request with the supervisor.
- Dataset release: the data were collected by the thesis author and B. Danilovaitė together; cite her thesis and agree any public release with her and the supervisor.

## If a new cohort is recorded (lessons from M0)
- **Back up raw files on the day**, read-only, with checksums. Analyse copies only, never re-save originals (127 files in this cohort were altered in 2020).
- **More dams, housed separately**: at least 6 litters from 6 dams. Assign treatment within litters (split-litter design) so drug and litter effects are separable.
- **Log per recording**: date and time, pup marking, order, weight, chamber temperature (a reliable thermometer), and who handled the pup. Use the same `X.X.X.X` marking scheme and record the birth date as P0.
- **Break the age–session link**: randomize recording order within a day, record ambient or empty-chamber control files each session, and record more than one litter per day where possible.
- Keep the same acquisition settings (Avisoft, 214285 Hz, 301 s, microphone at 10 cm), or document every change.
