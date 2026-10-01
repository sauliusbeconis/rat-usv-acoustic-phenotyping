# 0012 — M4 training settings fixed before full runs

Date: 2026-10-01. Fixed before any full-length M4 run.

- Encoders: AVA-style VAE (rung 1), small supervised CNN regressing pup age per call (rung 2), and a SimCLR-style contrastive encoder (rung 3). All share the same convolutional trunk and a 32-d embedding (`src/rat_usv/representations/`).
- Epochs: VAE 40, CNN 30, SSL 60. Adam lr 1e-3; batch 128 (SSL 256); bfloat16 autocast on the RTX 4070.
- Seeds 0, 1 and 2 for every rung; all runs are reported.
- Evaluation as in protocol v1 (leave-one-litter-out, paired pup bootstrap, background-embedding control, refit without `edit_suspected`).

**Disclosure.** 1-epoch smoke runs of each rung were executed on the real folds to test the code, and their held-out numbers were seen. They are not results. No setting above was chosen or changed in response to them. The runs are kept in `.local/runs/*-m4-{vae,cnn,ssl}-*` from 2026-10-01 ~13:20 UTC.
