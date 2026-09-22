# M0.1 evidence extraction pilot

M0.1 reconstructs observations and historical assertions. It does not confirm animals, litters, dams, sex, age, acquisition units or recording roles. Protocol v0 remains a draft. Final splits, training, collection-wide decoding and a full source freeze are later milestones.

## Run a bounded selection

Use `uv run --frozen usv evidence-pilot --config /path/to/config.local.yaml --selection /path/to/selection.local.yaml`. All outputs go to a new run below the configured output root. Local selections, inventories, reports and source excerpts must remain ignored. The source/output separation rule also applies to explicitly selected inputs: no excluded subtree, symlink or traversal can be selected as source.

Source roots and output roots resolve from the config YAML. The selection's protocol and optional baseline inventory resolve from the selection YAML. Artifact paths are relative to the source root. This synthetic example describes the selection format:

```yaml
schema_version: 2
protocol: ../../protocol/SCIENTIFIC_PROTOCOL.md
max_source_bytes: 1073741824
sources:
  - path: example/0001.wav
    kind: wav
    lineage: independence_unreviewed
    selection_reason: explicitly selected synthetic example
  - path: example/export.xlsx
    kind: workbook
    lineage: historical_export
    selection_reason: candidate export; correspondence unconfirmed
  - path: history.zip
    kind: archive
    lineage: historical_export
    selection_reason: inspect directory and selected members without extraction
    members: [example/export.xlsx]
candidate_links:
  - recording: example/0001.wav
    export: example/export.xlsx
    reason: candidate for interval comparison, not biological identity
unresolved_decisions:
  - Confirm original acquisition and export units with the researcher.
```

Other source kinds are `script` (UTF-8 R/Rmd text only) and `pdf` (explicit positive, one-based PDF `pages`). PDF extraction needs Poppler `pdftotext` on PATH; its version is recorded. There is no R runtime execution, workspace restoration or historical object deserialization. Synthetic CI does not need Poppler or private sources.

Optional `statements` specify source `path`, exact `locator`, a required `contains` text anchor, candidate `subject`, `field`, typed `value`, optional `original_units` and `context`. These are unconfirmed analyst interpretations supported by source text. They do not inherit the epistemic status of the source's author. PDF text extraction does not transcribe figures. A text anchor establishes source location, not truth or applicability to this collection.

## Evidence and migration

`evidence.json` uses schema version 2. The private v1 Parquet registries remain unchanged. This is an additive sidecar migration into a new run, never an in-place rewrite of a historical manifest. No command promotes this pilot into a canonical registry.

| Record | Meaning |
| --- | --- |
| Artifacts | Source-relative path, whole-file/member SHA-256, size, declared lineage; archive members also have container SHA, member name and ordinal. Filename equality is insufficient to identify a member. |
| Observations | Stable content ID, artifact ID, parser version, exact locator, raw typed JSON and manifest digest. Raw numeric lexemes remain strings; zero, underscore, stored empty text, explicit blank and absent cells remain distinguishable. |
| Assertions | Scoped candidate subject, typed value, original units, context and supporting observation IDs. Weights require a measurement occasion. Historical day labels remain labels, never automatically postnatal age. |
| Relationships | Stable edge ID, observation endpoints, supporting assertion IDs, alternative edges, evidence class and review history. Byte-identical edges establish artifact equivalence only. |
| Reviews | Actor, UTC timestamp, protocol digest, supporting/opposing records, explicit competing dispositions and supersession. Rejection/supersession retain prior records. |

`uv run --frozen usv validate-evidence --bundle /path/to/run/evidence.json` checks typed records, reference integrity, observation content IDs, manifest digest and artifact correspondence. It checks review prerequisites, conflicts in the same subject/field/context, supersession cycles and derivation subjects. STRONG confirmation requires distinct content hashes, distinct lineages and explicit independence reviews. Neither distinct files nor copied scripts constitute independent corroboration automatically. Validation never certifies scientific readiness.

The v1 validator additionally checks JSON syntax, SHA-256 syntax, supersession references/cycles, evidence-to-subject agreement and animal/litter dam agreement. Version 1 cannot express the full contextual review model; use v2 for new review work.

## Parser coverage and limits

- Workbook package reads are bounded; external worksheet targets, DTD/entities, duplicate coordinates, ambiguous package members and invalid indexes fail explicitly. Shared strings, styles, formula text/attributes and caches are retained separately. No calculation or unit conversion occurs. Missing cells are enumerated only within A1 through the maximum explicitly stored cell, with a rectangle budget. Summary labels `min/max/mea/std/cnt` are separated from numbered event candidates.
- WAV inspection seeks over audio payloads, records chunk offsets/sizes and preserves metadata bytes. `TIME` strings retain unresolved date convention; `bext` fixed-field offsets follow [EBU Tech 3285 v2, section 2.3](https://tech.ebu.ch/docs/tech/tech3285.pdf). Private `LBL ` lines retain exact bytes; numeric intervals are candidates, not validated annotations. Unknown chunks remain directory observations. Unsupported/truncated/oversized structures fail explicitly. Header success is not full decode QC.
- Complete ordinal label/export comparisons retain every event row, boundary delta and unpaired label. The milliseconds/seconds hypothesis is explicit; no tolerance promotes a candidate link or merges animals.
- Script parsing is a deliberately narrow lexical extractor for explicit sex filters and quoted date recodes inside R code. Every line and original bytes remain available for review; broader R semantics are not interpreted. Unquoted numeric aliases remain exact tokens and receive leading-zero risk entries.
- Archives are never extracted. Directory entries retain ordinals, byte offsets, duplicate-name counts and unsafe-path flags. Selected safe members are read by ordinal under member/aggregate budgets, hashed independently and parsed in memory. Copied observations remain separate.
- Each selected outer source is fully hashed before and after extraction with signatures. A source change, baseline mismatch or parser error makes the run unsuccessful; failed runs and partial observations remain. Limits are 64 selected files, at most 512 MiB per outer file, configurable total up to 2 GiB; package metadata limits are 8 MiB/member, 64 MiB aggregate, 10,000 entries and 200,000 worksheet cells. Full source hashing necessarily reads WAV payload bytes, but the metadata parser does not decode them.

## Review bundle

Start with `review.md`, `coverage.json`, `conflict_ledger.json` and `interval_comparisons.json`. `recording_evidence.json` indexes acquisition, chunks, labels, candidate exports, unknown biological hierarchy and zero decode coverage. `source_manifest.json` and `verification.json` document bounded source immutability, not a dataset freeze. `run.json`, `code_snapshot.json`, `selection.json`, `protocol_snapshot.md` and the emitted schema preserve the run's inputs and implementation. Selected WAVs are logged as prior exposure; this does not determine eventual test eligibility.

The pilot's selector records the exact named example and lexicographic choices for the three specified anomalous folders, whose exact WAV filenames were not supplied. Candidate exports for anomalous folders remain competing hypotheses. Private bundle counts include dependent copies and must never be reported as independent animals or confirmations.

## Engineering acceptance

Run the five frozen-uv checks in AGENTS.md, then `uv run --frozen python scripts/check_source_visibility.py`. That script verifies every source module and the CI example config are Git-visible, copies the proposed visible files into a disposable repository, runs the data guard on its temporary index and imports all modules from an exported clean tree. It never stages or commits the real repository. Because the earlier scaffold is untracked, this is a candidate-checkout check; the existing HEAD does not yet contain the implementation.

Remaining scientific decisions are recorded privately: identity continuity and anomalous groups, birth/day conventions, original weight/sex/dam/mortality/allocation records, acquisition/edit history, access/reuse rights, backup and source quiescence, independent annotators, endpoint/estimand and protocol review. Missing evidence blocks only the affected scientific step.
