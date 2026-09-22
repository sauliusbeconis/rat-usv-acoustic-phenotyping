# Visualization strategy

Three distinct purposes have separate output directories and explicit labels.

| Category | Examples | Evidence requirements |
|---|---|---|
| scientific | Dataset flow/coverage; animal trajectories; effect/uncertainty plots; baseline/ablation comparisons; calibration; real-call morphology | Registered figure ID, claim/run/data/config/split hashes, independent N, uncertainty and reproduction script |
| exploratory | PCA/UMAP/t-SNE, call neighborhoods, linked spectrogram/contour/audio explorer, missingness maps | Seed/parameters/fit scope, projection distortions and missing metadata visible; no confirmatory claims |
| presentation | Conceptual NASS architecture, illustrative 3D/topographic acoustic landscape | Label schematic or projection; do not imply quantitative geometry unsupported by analysis |

Any interactive call-space should reveal actual source path/interval, known animal/day/litter, mapping/evidence status, preprocessing and recording domain; unknowns remain visible. Export reviewed de-identified versions only. Every click-through exemplar must be recoverable from source hashes and time offsets. Separate real waveforms from frequency-shifted listening aids.

Use consistent units, colorblind-safe palettes, readable fonts, vector scientific outputs where possible and honest axes. Show individual independent units rather than hiding them under smooth means. Projections and smooth latent trajectories need quantitative support outside the visualization. Different seeds, dimensionalities and neighborhood settings test projection stability; fit exploratory projections on permitted data and disclose exposure.

`figures/figure_registry.csv` is initially header-only. Output convention: `reports/figures/{scientific,exploratory,presentation}/FIG-ID_description`. Review generated figures for private labels before Git/publication. The implemented dataset-atlas currently emits technical JSON/Markdown; interactive and paper figure generators are future work.
