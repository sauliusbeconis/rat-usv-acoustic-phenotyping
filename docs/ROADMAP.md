# Roadmap

This work is a **bachelor's thesis** (Vilnius University, Department of Neurobiology and Biophysics; supervisor Osvaldas Rukšėnas). M1–M4 form the thesis core; M5 and M7 strengthen it; M6 (NASS) is optional and only justified by an M4/M5 gap. M8 (external / perturbation validation) uses **public datasets only**; no new experiment will be run.

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
| M8 | External / perturbation validation (H4) | Public rat pup datasets (control and disease-model or drug groups) processed by the frozen M1–M4 pipeline, with no refitting on their animals | Representation and trajectory tested on the external cohort; group displacement compared with classical measures; null results recorded. Where only call-feature tables are public, the test runs at the classical-feature level | Candidates listed below |

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

## External data (M8 candidates; owner-supplied list, checked 2026-10-01)
| Dataset | Verified content | Use |
|---|---|---|
| **Fragile X rat pups**, Fmr1 KO vs WT ([Zenodo 20940043](https://zenodo.org/records/20940043), CC BY 4.0) | *Checked:* one DeepSqueak call table, 9,495 accepted calls, 33 pups (17 WT, 16 KO), P3/P6/P10/P14/P21 (not every pup on every day; only 8 and 6 pups have calls at P14 and P21). Per call: begin/end time, 14 DeepSqueak features, call label, genotype. **No audio.** Some pup IDs (e.g. `lb`) may be reused across litters and need checking against file names | **Primary H4 test** at the classical-feature and timing level: transfer our developmental model to WT pups and test KO displacement. Requires our M2 features to include DeepSqueak-equivalent definitions |
| **USCMed** adult male rats, fear context ([Zenodo 15029872](https://zenodo.org/records/15029872), open, 5.95 GB zip; later versions restricted) | *Checked:* open raw-audio archive with hand-scored annotations (27 rats) | M1 detector validation on independent raw audio; robustness of representations to another lab and recording setup |
| **mouseTube rat benchmark**, Rat 16 Conditioning ([Zenodo 20626750](https://zenodo.org/records/20626750), CC BY 4.0) | *Checked:* one 90 MB WAV (300 kHz, Avisoft) with two independent manual annotations plus outputs from Avisoft, DeepSqueak, RatRec, SonoTrack, USVSEG and the LMT USV Toolbox | M1 detector benchmark and inter-annotator agreement reference |
| Mahler Lab NAc amphetamine + DREADD/CNO (GitHub `USVs.csv`, ~77k calls) | *Not yet checked:* reported as a DeepSqueak call table with manual categories; adult Long-Evans; no WAVs | Feature-level pharmacology (adult 50-kHz calls); outside the developmental core |
| Banjac et al. 2026, amphetamine/morphine in WKY vs Wistar (Uni Ljubljana repository, CC BY 4.0) | *Not yet checked:* reported as XLSX tables, no WAVs | Feature-level drug × disease-model test in adults; outside the developmental core |
| Shank3 rats, TrackUSF (Mendeley Data 10.17632/8d4yz5fyhy.2, CC BY 4.0) | *Not yet checked:* reported as processed TrackUSF outputs | Disease-model phenotype space in adults; low priority |

Adult pharmacology datasets measure a different call type (adult 50-kHz calls versus pup isolation calls), so transferring a pup developmental state space to them is a separate, weaker claim. They are optional for the thesis.

## Owner decisions still open
- **M8 data choice:** the Fragile X table is the primary external test; decide with the supervisor whether to ask its authors (Auerbach lab, UIUC) for audio, which would allow representation-level transfer.
- **Model design (M4–M6) is an explicit owner decision.** Agents propose options with trade-offs and wait for the owner's choice before implementing any model architecture.
- Dataset release: the data were collected by the thesis author and B. Danilovaitė together; cite her thesis and agree any public release with her and the supervisor.

