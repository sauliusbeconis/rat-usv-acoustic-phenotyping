# M0.4 bounded metadata discovery

`metadata-search` locates candidate animal-marking, birth and recording-session evidence in an explicitly selected, hash-bearing inventory. Matches are observations for review, not verified identities, dates, ages or independent animals. Original logs, retrospective descriptions, analysis code and filenames must remain distinguishable.

```bash
uv run --frozen usv metadata-search \
  --config /path/to/config.local.yaml \
  --inventory /path/to/inventory/source_inventory.parquet \
  --policy /path/to/search_policy.local.json
```

The config selects the source root and an output root separated from source traversal. All inventoried files must have hashes. Policy patterns are analyst-supplied regular expressions, for example:

```json
{
  "patterns": {"birth": "birth|born", "marking": "marking", "session": "recording|session"},
  "max_files": 3000,
  "max_file_bytes": 16777216,
  "max_source_bytes": 134217728,
  "max_records": 100000,
  "max_pdf_pages": 300
}
```

Keep actual policies, inventories, excerpts and reports private. Choose language-appropriate patterns and review context; a keyword can occur in unrelated literature or executable code. No match does not establish that a record is absent.

## Outputs and coverage

- `entities.json` records every selected file and ZIP member, hashes, member ordinals, inspection status and workbook sheet/header coverage. Duplicate copies remain separate.
- `matches.jsonl` preserves exact text, line/cell/page locators and raw cell representations, including separate formulas and caches. Numeric aliases are never coerced. Match IDs join to entity IDs.
- `pdf_text/` contains extracted page text, linked through `pdf_text_manifest.json`. Poppler is required for PDFs; its version is recorded in the PDF entity. Figures require separate visual inspection.
- `verification.json` compares each source's hash and filesystem signature before and after inspection. `failures.json` retains errors; operational failures produce an unsuccessful summary and CLI exit status.
- `summary.json`, `policy.json`, `input_inventory.json`, `code_snapshot.json` and `run.json` document counts, bounds, inputs, implementation and output hashes. Success means no operational failure, not complete coverage or scientific readiness.

XLSX parsing reads package metadata and the stored cell rectangle under the existing bounded parser. It never calculates formulas. Text uses UTF-8 or explicitly marked UTF-16; CSV lines remain original strings. HTML text excludes script/style tags without rendering or fetching. Historical R/Rmd is only text. Serialized R objects, images, embedded objects and unsupported formats remain explicit gaps. ZIP entries are read by ordinal in memory, never extracted; nested archives and PDF members are not recursively parsed.

Existing package bounds remain 8 MiB/member, 64 MiB uncompressed aggregate, 10,000 entries and 200,000 cells/workbook. File count, outer-file bytes, retained matches and PDF pages have additional policy bounds. A failed or partial run is retained and must not be described as an exhaustive successful search.

## Review and acceptance

After discovery, use an explicit `evidence-pilot` selection to retain relevant full script lines and PDF pages. Add anchored historical statements, preserve conflicting assertions and validate the resulting bundle. Derivative birth tables and session indexes must link back to assertion/observation IDs, source hashes and the versioned protocol. A marking legend cannot establish individual continuity; a script day label cannot establish postnatal age.

The private milestone handoff is indexed at `.local/setup/m04_run_index.json` after validation. It records recovered evidence, source-specific coverage and remaining decisions. Original records can be reported as unlocated within inspected sources, with explicit gaps; unavailable formats cannot establish global absence.

Run the five frozen-uv checks from AGENTS.md and the candidate-checkout source visibility check. Synthetic tests cover immutability, leading zeros, formulas/caches, duplicate archive members, unsafe member paths, unsupported encodings, failed-run retention, output reuse, limits and PDF text provenance without requiring private data or Poppler in CI.
