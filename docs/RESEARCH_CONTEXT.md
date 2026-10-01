# Research context (owner's brief)

This is the authoritative statement of the project's scientific aims, written by the project owner. `AGENTS.md` summarizes it; when they differ, this document wins on *what* to study, `AGENTS.md` on *how* to work.

---

PROJECT CONTEXT — RAT USV ACOUSTIC PHENOTYPING

This project studies rat ultrasonic vocalizations (USVs) as a potentially rich functional readout of biological state.

The central idea is that conventional USV analysis throws away a large amount of information. Traditional analyses usually reduce vocalizations to a small number of summary variables such as call count, duration, mean/peak frequency, bandwidth, or coarse 22-kHz / 50-kHz categories. Our hypothesis is that the full spectrotemporal structure of vocalizations — together with their temporal organization into bouts and sequences — contains substantially richer information about development and functional nervous-system state.

The primary biological question is:

Can multidimensional acoustic structure of rat USVs capture biologically meaningful changes in postnatal development more completely than conventional acoustic summary variables?

The initial biological system is longitudinal postnatal development. We want to determine whether development produces a coherent, measurable trajectory through a multidimensional "acoustic state space" rather than merely independent changes in a few acoustic variables.

A useful conceptual formulation is:

    animal biological state
            ↓
    vocal production system
            ↓
    USV morphology + timing + sequence structure
            ↓
    measurable acoustic phenotype

We therefore want to reconstruct a multidimensional acoustic phenotype from USVs and study how that phenotype changes over developmental time.

This should NOT be treated as an ordinary age-classification problem.

Predicting postnatal day may be useful as a validation task, but the scientific target is the underlying structure of the developmental acoustic phenotype, its dynamics, its biological interpretation, and its generalization.

A strong result would not merely be:

    "the model predicts P14 with high accuracy"

but something closer to:

    "postnatal development follows a reproducible trajectory in acoustic representation space; this trajectory contains information beyond conventional USV variables; the effect is consistent across held-out animals; and the learned changes can be traced back to identifiable acoustic and temporal properties."

## Core hypothesis

Primary hypothesis:

Multidimensional USV acoustic structure contains information about postnatal developmental state beyond conventional acoustic summary variables.

Possible secondary hypotheses:

1. Developmental change is continuous or structured as a trajectory through a latent acoustic state space rather than a set of independent univariate changes.
2. Temporal organization of calls — inter-call intervals, bouts, transitions, repertoire structure, sequence entropy, etc. — contains additional biological information beyond individual-call morphology.
3. A representation learned from USVs may capture biologically meaningful structure that transfers to unseen animals and, eventually, to independent datasets or experimental contexts.
4. Functional perturbations such as pharmacological CNS modulation may cause measurable displacement in the same or related acoustic state space, potentially even when conventional USV summary metrics change little.

The latter points are future validation/generalization directions, not assumptions that should be treated as established results.

## Scientific positioning

This is fundamentally a neurobiology / biophysics / computational bioacoustics project. Machine learning is an analytical instrument, not the scientific endpoint. We are not trying to build "another USV classifier". The scientific object is the relationship between biological state and high-dimensional acoustic output.

USV production should be treated as an integrated functional output involving central neural control, respiratory patterning, vocal motor control, and the resulting acoustic signal. However, the project must NOT infer specific neural mechanisms from acoustic data alone.

Use language such as: acoustic phenotype; functional acoustic marker; acoustic state; developmental trajectory; multidimensional acoustic representation; spectrotemporal structure; temporal organization.

Avoid unjustified claims such as direct neural-state decoding, proof of a specific neural mechanism, or causal biomarker — unless the experimental evidence eventually supports them.

## Model ambition

The modeling component should be unusually strong. We do not want to stop at spectrogram → ResNet → classifier. The long-term goal is to investigate an advanced, biologically motivated architecture that may itself become a methodological contribution.

Working internal concept: **NASS — Neuroacoustic State-Space model.** This is a working name only. Do NOT claim novelty until a formal literature review demonstrates it.

A possible architecture may include:

    raw high-sample-rate USV audio
            │
            ├── high-resolution spectrotemporal branch
            │
            └── explicit acoustic morphology / contour branch
                        │
                        ▼
                  call representation
                        │
              self-supervised learning
                        │
                 bout / sequence model
                        │
                 animal-day state
                        │
           longitudinal latent state-space
                 P1 → P2 → ... → P21

A hierarchical latent structure may conceptually look like `z_call → z_bout → z_session → z_animal(t)`. We are also interested in separating useful biological variation from nuisance variation: `z_observed ≈ z_biology + z_individual + z_technical`.

Potential future modeling ideas include: biologically constrained self-supervised learning; multi-view acoustic representation learning; masked acoustic modeling; contrastive learning; hierarchical sequence models; transformers or state-space sequence models where justified; continuous-time / longitudinal latent models; variational state-space approaches; nuisance-invariant or domain-adversarial learning; concept bottlenecks; prototype-based explanations; disentangled representations; uncertainty-aware latent states.

However, architecture complexity is not a goal by itself. Every component must answer a scientific need:

- A sequence model is justified if call order contains useful biology.
- A longitudinal model is justified because the same animals are repeatedly observed through development.
- A nuisance-invariance component is justified if recording/session/background signals could contaminate the biological representation.
- A multi-view architecture is justified if explicit acoustic morphology and learned spectrotemporal information contribute complementary information.

The advanced model must earn its complexity by outperforming or revealing information beyond simpler baselines.

## Model comparison ladder

    classical descriptive statistics
      → predefined acoustic features
      → regularized linear models
      → tree-based ML
      → simple spectrogram CNN
      → self-supervised acoustic representation
      → hierarchical bout/sequence representation
      → longitudinal acoustic state-space model

All comparisons must use compatible biological evaluation splits. A highly complex model that adds no meaningful information beyond a simple model is not scientifically superior.

## Explainability

Explainability should be built into the research design rather than added as a cosmetic post-processing step. The eventual goal is to trace a result through several levels:

    developmental / functional state change
      → learned latent factor or trajectory
      → specific call families / bouts / sequence patterns
      → physical acoustic characteristics
      → representative real calls
      → spectrogram / contour / audio example

For important results we should be able to ask: Which animals drive this effect? Which developmental days? Which calls or bouts? Which acoustic structures contribute? Is the effect consistent across animals? Across litters? Robust across model seeds? Could background/session/technical variables explain it? Does the interpretation survive alternative model architectures?

## Visualization is a major part of the project

Three clearly separated classes:

1. **Scientific / evidence figures** — publication-defensible: individual-animal longitudinal trajectories, confidence intervals, effect sizes, classical feature trajectories, distance matrices, held-out predictions, calibration, model comparisons, ablations, negative controls, litter/animal-level uncertainty. These support scientific claims.
2. **Exploratory scientific visualization** — PCA, UMAP, latent-space maps, density maps, cluster topology, acoustic repertoire maps, sequence transition graphs, developmental trajectories, embedding stability plots. These may generate hypotheses but cannot by themselves establish biological separation.
3. **Conceptual / presentation visualization** — unusually strong visual explanation: a 3D "acoustic landscape"; aerial/topographic views of latent state space; developmental trajectories moving through it; density surfaces of repertoire; animated P1 → P21 development; an interactive latent space where clicking a point reveals animal, day, spectrogram, contour and real audio; graphical abstracts. These can be visually ambitious and novel, but must be clearly distinguished from statistical evidence.

Guiding metaphor: **development as a trajectory through an acoustic landscape** — the landscape is the learned acoustic state space; individual animals follow trajectories through it. The metaphor must be connected to real quantitative results rather than used as decoration.

## Scientific rigor

Assume the work may be submitted for publication and examined by bioinformaticians, neurobiologists and biophysicists who actively look for leakage, pseudoreplication, confounding and overinterpretation. Therefore: scientific validity > model performance.

- The biological experimental unit is the animal, not the vocalization. Thousands of calls from one animal are not thousands of independent samples.
- Respect repeated measures, litter structure and other hierarchical dependencies.
- Primary evaluation uses unseen animals; where possible, unseen litters. The strongest eventual evidence is independent cohort / independent laboratory validation.
- Random call-level splitting is unacceptable for primary claims.
- All learned preprocessing respects train/test boundaries: normalization, feature selection, dimensionality reduction, self-supervised training (for strong unseen-animal claims), augmentation tuning, hyperparameter optimization.

## Negative controls / adversarial validation

We should actively attempt to prove our own result wrong. Reviewer attacks to turn into explicit experiments:

- "The model learns recording date, not development."
- "The model recognizes the animal." / "…the litter."
- "The model recognizes microphone/background conditions."
- "The apparent clusters are just UMAP artifacts."
- "The deep model only wins because it received more tuning."
- "The sample size is actually the number of animals, not the number of calls."
- "The model is exploiting amplitude or SNR."
- "The result disappears outside this laboratory."

Examples: predict developmental day from background-only audio; from non-USV spectral regions; test whether session/date can be decoded; whether animal identity or litter identity dominates the representation; label-permutation controls; duplicate detection; representation stability; ablations; nested grouped cross-validation; comparison in the original high-dimensional representation rather than only projected UMAP space.

If a negative control succeeds unexpectedly, treat it as a discovered confound, not as an interesting model result.

## Reproducibility

Every important result should be traceable: raw recording → preprocessing configuration → detected/segmented calls → representation → model run → statistical analysis → figure → scientific claim. Each major experiment records: Git commit, dataset version, split definition, preprocessing config, model config, random seed, software environment, hardware, metrics, predictions, figures, interpretation, limitations.

Notebook-only results are not authoritative. Notebooks may be used for exploration; useful logic migrates into tested source code and reproducible workflows.

## Claim discipline

Claims move through evidence states: HYPOTHESIS → OBSERVED → REPLICATED → INTERNALLY_VALIDATED → EXTERNALLY_VALIDATED → SUPPORTED / REJECTED. Do not write conclusions first and then search for evidence. Do not describe exploratory results as confirmed findings. Do not describe correlations as mechanisms.

## Failed results matter

Null results, failed models, discovered confounds, unstable representations and abandoned hypotheses are preserved in the research record. Do not rewrite the project's history to make the final result appear inevitable.

## Publication standard

Another technically strong group should be able to inspect the repository and see: biologically correct experimental reasoning; leakage-aware evaluation; clear provenance; reproducible analysis; explicit negative controls; sensible statistical inference; advanced but justified modeling; interpretable results; visually excellent communication; clear separation between evidence and conceptual visualization; an honest record of uncertainty and limitations. It should feel like a small, rigorous computational research program, not a student ML exercise.

## Short version

Rat USVs are not merely counts or average frequencies. They form a high-dimensional acoustic phenotype whose morphology, timing and sequence structure may change systematically with biological state. Postnatal development provides the first test case. We want to learn and validate this acoustic state space, understand its developmental trajectory, determine what information it contains beyond classical USV variables, explain which acoustic structures produce the effect, and ultimately test whether the learned representation generalizes beyond the original animals and experimental context.

AI is the instrument. The biological structure of USV is the subject.
