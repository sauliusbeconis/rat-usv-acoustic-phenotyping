# Next steps

**Current milestone: M5 temporal organisation** (M7 first pass done: decisions 0015, 0016; M5's sequence model needs an owner design decision) (M4 incl. H2/H2b done: decisions 0013, 0014) (see `docs/ROADMAP.md`).

M4/H1 done (decision 0013): learned call embeddings add age information beyond classical features under leave-one-litter-out (combined with classical: VAE +0.66 d, CNN +1.04 d over classical alone; SSL not credited). Runs: `.local/runs/m4_overnight_table.json`, `.local/runs/m4_rich_baseline_secondary.json`.

1. **H2 (protocol estimand):** trajectory consistency of held-out pups in representation space. Use the VAE (unsupervised; age never seen in training) as the primary space and the CNN as secondary: the age ordering of animal-day centroids, the distance–age relationship, compared with univariate classical trajectories.
2. **H2b:** does the acoustic state track body weight beyond age? Use within-age, animal-level models on embeddings and classical features.
3. **M7 explanation chain:** which call types or acoustic properties drive the learned gain. Show nearest real calls and spectrograms per latent direction, and build an exploratory explorer.
4. **M5:** temporal organisation (bouts, transitions); start with order-shuffle ablations.
5. Before M8: verify the DeepSqueak-equivalent features; the Fragile X test is feature-level only.
