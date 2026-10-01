# 0011 — M4 primary split and model design (owner decisions)

Date: 2026-10-01.

1. **Primary split:** the owner chose co-primary "leave-one-litter-out + session-blocked leave-one-animal-out". Every littermate is recorded in every session of its litter, so the session-blocked variant removes the held-out pup's whole litter from training; it is identical to leave-one-litter-out. Implemented as leave-one-litter-out scored per pup (3 folds, 28 test pups). Leave-one-animal-out is reported only as a leak-inflated upper bound.
2. **Model design:** the full ladder. AVA-style VAE baseline, then a small supervised CNN, then a contrastive self-supervised encoder. Each is trained per fold on the training litters only, embeddings are pooled per animal-day, and they are compared with and combined with classical features using the M3 decoder and controls.
3. **Protocol v1 locked** before any M4 held-out result was seen (`protocol/LOCK.sha256`).
