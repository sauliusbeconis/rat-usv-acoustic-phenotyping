# Staged analysis plan

| Stage | Work | Gate / deliverable |
|---|---|---|
| M0 | Source freeze, identity reconstruction, acquisition/QC, historical spreadsheet and R provenance | Reviewed registries, unresolved report, technical/biological atlas and protocol v1 proposal |
| M1 | Reproduce classical acoustic summaries and historical analyses where traceable | Animal-level trajectories, hierarchical uncertainty, discrepancies documented |
| M1.5 | Negative controls and leakage attack suite | Background-only, date/session, amplitude/noise, litter, duplicate and shuffled-label probes |
| M2 | Linear/tree, CNN and SSL representation benchmarks | Same frozen biological holdouts, full tuning budget and fit scope |
| M3 | Conditional NASS hierarchy/state-space research | Benefit beyond simpler baselines, ablations, uncertainty and identifiability limits |
| M4 | Explanation chain and registered scientific figures | Real-call provenance and quantitative support; exploratory views labelled |
| M5 | Locked external validation appropriate to claim | Cohort/domain applicability, calibrated uncertainty, qualified claim status |

Current delivery implements M0 tools and governance only. No final split, detector, feature extraction or model training has been run.

M0 sequence: inventory → inspect source formats and duplicate copies → assertion ledger → candidate crosswalk → human resolution → acquisition/recording registry → QC review/exclusion ledger → Dataset Atlas → protocol lock proposal. Read historical code without executing it; record exact worksheet/cell or script/line references, formula versus cached value, and any transformations of aliases/days.

Planned inference: specify endpoint-specific hierarchical models after design reconstruction. Repeated animal observations need appropriate random effects/correlation or animal/litter-aware resampling. Report independent N and number of observations separately. Avoid treating thousands of calls as thousands of animals. Assess residuals, sensitivity to unresolved identities and exclusions, missingness and imbalance. Do not use last-observation imputation without scientific justification. Bootstrap and permutation units must preserve the experimental hierarchy and label exchangeability.

Model comparison: grouped nested tuning within development data; one reviewed outer biological split plan shared across methods. Primary inductive SSL cannot learn from test animals, even without labels. Use simpler nuisance-only and acquisition-only models as controls. Predefine primary metric and confidence interval before final evaluation. Preserve all seeds and failed configurations. A favorable projection or one seed is insufficient evidence.

M0 stop points: conflicting aliases, uncertain chronology, incomplete litter/dam links, inseparable batch and age, missing acquisition calibration, unknown exclusion rules. Record what evidence would resolve each. Claims must be weakened where resolution is impossible.
