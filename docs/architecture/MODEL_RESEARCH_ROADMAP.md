# NASS — Neuroacoustic State-Space research roadmap

Status: **working internal concept**, not implemented, not a validated neural-state model and not an established novelty claim. Each added mechanism must address an evidenced limitation and survive an ablation. Complexity is conditional on independent animal/litter count and identifiability, not the number of calls.

## Benchmark ladder

1. Classical summaries and hierarchical statistics: count, duration, frequency, bandwidth, intervals, animal trajectories and litter effects, with validated definitions.
2. Regularized linear models on reviewed acoustic features; transparent baseline for incremental information.
3. Tree ensembles with the same grouped splits and tuning budget; test nonlinear feature interactions.
4. Small spectrogram CNN; measure what raw spectral patterns add over features.
5. Self-supervised encoder, frozen linear probe before fine-tuning; determine label efficiency and transfer under train-only exposure.
6. Hierarchical sequence model and longitudinal state-space extension, only if simpler approaches leave a reproducible temporal-information gap.

Same biological partitions, primary endpoint, uncertainty procedure and data access scope across every comparison. Lock final test access; grouped inner folds for tuning. Count seeds, failures and compute budgets. An RTX 4070 12 GB is a resource constraint, not an argument for a particular model. Start with profiled bounded windows and modest encoders after M0; no training configuration is implemented now.

## Mechanisms as falsifiable proposals

| Mechanism | Scientific reason to investigate | Required comparator / falsifier | Main risk |
|---|---|---|---|
| High-rate audio / spectral view | Preserve measured ultrasonic timing and spectral structure | Reviewed frequency coverage; feature baseline; matched resolution ablation | Aliasing, acquisition response, noise, amplitude confounding; do not reuse speech-rate preprocessing blindly |
| Explicit contour/morphology branch | Test whether pitch trajectory, slope, curvature, jumps and duration add interpretable information | Spectral-only, contour-only and fused branches; perturb contour extraction/annotation | Detector/tracker error can masquerade as developmental morphology |
| Multi-view fusion | Complementary representations may capture different acoustic properties | Parameter-matched single view; missing/noisy-view robustness | Gains may only reflect capacity or technical shortcuts |
| SSL representation | Use unlabelled training acoustics while reducing dependence on arbitrary call categories | Frozen linear probe, supervised small CNN, random encoder and augmentation ablations | Biological signals erased by augmentations; test-animal exposure; false negatives from related calls |
| call → bout → session → animal-day | Model dependence and temporal organization at observed biological levels | Mean pooling, matched call counts, shuffled within-session order, removed bout boundaries | Threshold choice, uneven counts and session confounding |
| Longitudinal latent z_animal(t) | Encode repeated observations and uncertain change over evidenced age/time | Mixed-effects/spline baseline, static pooled representation, shuffled time | Sparse irregular sampling, smoothness prior imposing an artificial developmental trajectory |
| z_biology + z_individual + z_technical | Explore whether documented sources of variation can be separated | Identifiability analysis, supervised nuisance probes, intervention-free sensitivity and held-out batches | Components are not identifiable merely because they are named; age and batch may be inseparable |
| Nuisance-invariant/domain-adversarial learning | Test robustness to measured acquisition domains | No-adversary baseline; domain probe; biological endpoint retained; batch holdout | Suppression may remove biology when nuisance and biology covary |
| Latent-to-real-call explanation | Link predictions to measurable acoustic properties and exact observed calls | Prototype/neighbor stability, perturbation tests, expert blind review | Attention/saliency alone is not faithful or causal explanation |
| Predictive/latent uncertainty | Separate noisy observations, missing times and model instability | Calibration, coverage under animal/batch holdouts and simple uncertainty baseline | Visually smooth trajectories can hide unidentifiable uncertainty |

Proposed state model: observed call/bout/session representations inform z_animal(t); transition structure and observation noise must be specified after sampling schedules are known. Assess filtering versus smoothing separately: a model using future sessions cannot support a prospective prediction claim at earlier time. Any developmental axis must be validated in original feature/representation space, not just a projection.

## Explanation chain

A supported developmental comparison → reproducible latent association → contributing bout/call families → measured acoustic properties → real source-relative WAV intervals, spectrograms and contours. Each link carries uncertainty, run/manifest hashes and alternatives. Retain negative examples and failures. Audio playback for humans may need a separately labelled frequency-shifted derivative, never a replacement for raw high-rate audio.

## Literature and novelty gate

Starting methodological anchors, not a completed USV literature review:

- [Ganin et al., Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818): gradient reversal is a candidate domain-invariance mechanism. Its success elsewhere does not establish validity for this cohort.
- [van den Oord et al., Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748): a possible self-supervised temporal objective, to compare with simpler and non-contrastive alternatives.
- [Locatello et al., Challenging Common Assumptions in Unsupervised Disentanglement](https://arxiv.org/abs/1811.12359): assumptions and inductive biases matter; named latent subspaces are not evidence of biological disentanglement.

Before any novelty claim, run a dated systematic comparison of rat developmental USV methods, call morphology/contour models, bioacoustic SSL, hierarchical vocal sequence models, longitudinal latent models and nuisance controls. Record databases, queries, inclusion decisions, closest methods, differences and reproducible baselines. Neural mechanisms are not observed directly in these audio files; keep terminology operational.
