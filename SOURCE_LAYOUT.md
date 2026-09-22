# Source/output layout contract

Configure the immutable source root in ignored `configs/data/local.yaml`. The following layout is schematic; actual paths, directory names, counts and source findings belong in private setup reports.

```text
/path/to/immutable/source/          original artifacts; read only
/path/to/rat-usv-acoustic-phenotyping/
  configs/data/local.yaml          ignored source pointer
  .local/runs/                     private, append-only-by-convention run outputs
  .local/setup/                    private inspection reports and handoff indexes
  registry/*.parquet               ignored schema placeholders or reviewed derivatives
```

Source artifacts must never be moved, renamed or rewritten by the research workflow. Directory names and filename stems are observations, not confirmed animal/day/litter semantics. Preserve leading zeros in all aliases. Historical copies may overlap or disagree; retain them and do not choose a canonical copy without evidence.

If the repository is inside the source root, add its exact source-relative path to `exclude_relative` in the local config. This prevents self-inventory and traversal of caches and generated outputs. Output validation rejects unexcluded descendants of the source root. Do not link raw sources into the repository with symlinks.

Future compute storage can use a separately provisioned output location outside the source tree, referenced only through ignored local config. Do not provision it by moving historical source artifacts. Establish an independent backup/snapshot policy before a full freeze.

Repository access is not permission to publish raw recordings, spreadsheet identities or thesis documents. Do not place source copies in Git or Git LFS. Code guard and ignore rules complement manual review.
