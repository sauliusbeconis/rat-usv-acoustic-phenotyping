# Next steps

**Current milestone: M4 — representation ladder, overnight training launched 2026-10-01 ~16:04 local** (decisions 0011, 0012).

The overnight chain (`.local/setup/overnight_m4.sh`, detached) runs VAE (40 epochs), CNN (30) and contrastive SSL (60) for seeds 0, 1 and 2 under leave-one-litter-out. Each run writes `.local/runs/*-m4-{kind}-*/m4_{kind}_summary.json`. When all 9 finish, the table is collected into `.local/runs/m4_overnight_table.json`.

In the morning:
1. Check `.local/runs/overnight_m4_*.log` for `ALL DONE` and any non-zero `exit`.
2. Read `m4_overnight_table.json`. For each rung and seed: per-pup MAE (classical, learned, combined), the primary gain (classical − combined) with pup-bootstrap CI, the background-embedding control, and the H1 criteria from protocol v1.
3. Write the M4 decision note: report all seeds, credit a rung only if it beats the earlier rungs and the classical features, and state H1 status. Then present results to the owner before M5.
4. Still open: verify the DeepSqueak-equivalent features (M8 prerequisite).
