# M0.5 historical metadata reconciliation

`reconcile-metadata` compares existing historical input references and date/day recodes against metadata-search derivatives. It opens no raw sources, executes no historical code and makes no biological assignments.

```bash
uv run --frozen usv reconcile-metadata \
  --config /path/to/config.local.yaml \
  --evidence /path/to/evidence-run/evidence.json \
  --search-entities /path/to/search-run/entities.json
```

The command snapshots the required upstream files into a new private output directory and verifies their recorded hashes. The evidence bundle, source manifest and protocol must agree. Evidence extraction must have succeeded; both upstream runs must have passed source-integrity checks and refer to the configured source root. Evidence source hashes must also agree with the search inventory. A partially successful keyword search is allowed only with its failure list preserved explicitly in the new summary. Local hashes are consistency checks, not independent scientific approval.

## Comparisons

- `input_references.json` retains narrowly recognized `Diena_N`/`diena_N` assignments to a quoted `read_excel` path. Candidates match the final three case-sensitive path components. Slash conversion is lexical only; no historical path is resolved or opened. Short paths, missing candidates and competing content hashes remain explicit. Byte-identical copies remain distinct entities with their archive ordinals.
- `mixed_input_contexts.json` flags different pairs of preceding path components under the same variable prefix within one script. These may be deliberate examples, reused variables or errors; the command cannot decide which.
- `day_recode_groups.json` preserves assertions by artifact, binding and recode block. Adjacent source lines are compared under a stated hypothesis: tokens represent `MM.DD` in forward order with at most one year rollover. Synthetic reference calendars cover leap-year alternatives; no real calendar year is assigned. Incompatible increments, non-increasing labels and invalid dates are flagged. Compatible arithmetic does not establish postnatal age or recording attendance.
- `dependency_lines.json` retains lexical package, I/O and execution-related call candidates for static review. It is not a complete R parser; strings, multiline expressions, dispatch, dynamic paths and package versions need separate review.
- `non_code_input_examples.json` preserves recognized-looking input lines outside supported R chunks. They are not silently discarded or treated as executed code. Original observations remain in the complete snapshotted evidence bundle.

All outputs retain exact evidence IDs, source locators and hashes. Counts include dependent copies and repeated assertions. No match promotes an alias, date, session or animal into a canonical registry. There is no automatic RData/RDS restoration.

## Bounds and validation

Each input snapshot allows at most 64 MiB/file and 128 MiB total; candidate edges are capped at 100,000. Required symlinked files, stale output hashes, incompatible source hashes and output-directory reuse fail explicitly. The CLI retains the failed run and error.

Synthetic tests verify source/derivative immutability, stale hashes, preserved duplicate archive members, leading zeros, ambiguous candidates, code/prose separation, copied recode scopes, calendar uncertainty and failed-run retention. Run the frozen checks in AGENTS.md and the source-visibility check before committing.

The private handoff is indexed at `.local/setup/m05_run_index.json` after validation. Any separately authorized binary-header preflight must preserve its own selection, byte bounds and source-integrity checks. Header recognition neither extracts object contents nor establishes that deserialization is safe or scientifically sufficient.
