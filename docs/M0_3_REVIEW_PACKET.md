# M0.3 evidence review packet

The review packet turns an existing M0.1 conflict ledger into material for source review. It opens only the supplied extraction derivatives. It does not reopen raw recordings, execute historical code, adjudicate conflicts or create scientific review decisions.

```bash
uv run --frozen usv review-packet \
  --config /path/to/config.local.yaml \
  --evidence /path/to/m01-run/evidence.json
```

The command writes a new directory under the config's output root. Config paths resolve relative to the YAML. Keep the output root private and excluded from source traversal. An existing output or an output within the input run is rejected.

## Validation and provenance

Eight upstream files are copied byte-for-byte into `inputs/`: `run.json`, `evidence.json`, `source_manifest.json`, `conflict_ledger.json`, `protocol_snapshot.md`, `coverage.json`, `failures.json` and `code_snapshot.json`. The fixed selection has limits of 64 MiB per file and 128 MiB total. Missing files, symlinked files and exceeded budgets fail explicitly.

Before preparing the packet, validate consumed file hashes against the upstream run, the typed evidence graph and source manifest, the protocol snapshot digest, and extraction/source-verification success. Ledger IDs must be unique, references must exist, and statuses must be `UNRESOLVED` or `CONFLICT`. The CLI retains failed attempts, their snapshots and error logs in new private runs. Hash consistency is an audit check, not independent approval or tamper-proof storage.

The packet's run record hashes its exact input snapshots and outputs and records the current code/config and lock hashes. This avoids assigning the provenance of a later-changed original to the bytes used for review. The upstream run and code snapshot are retained separately.

## Review outputs

| Output | Purpose |
| --- | --- |
| `review.md` | Every ledger item, its unchanged decision/status, assertion values and contexts, and source/member hashes with exact locators |
| `review_packet.json` | Every original ledger row expanded to complete referenced assertions, raw observations and artifact provenance |
| `assertions.json` | Every upstream assertion, including those absent from the ledger, with its source observations and locators |
| `relationships.json` | Every relationship, its alternatives and both source observation endpoints |
| `summary.json` | Coverage denominators, input hashes, missing evidence links and scope limits |
| `inputs/`, `run.json`, `code_snapshot.json` | Exact upstream bytes and provenance of packet construction |

Reference order, repeated references, aliases, original units, nulls, archive ordinals, dependent copies and competing links remain intact. A missing ledger reference is explicitly `missing_in_ledger`; the command does not infer a link by matching names. Record counts are not animal counts or independent corroboration. An empty ledger cannot establish scientific readiness.

The original ledger can contain gate descriptions that predate M0.2. Its wording is preserved as historical evidence; consult the newer handoff for bounded technical work already completed. A packet is a navigation aid, not a new `ReviewDecision` or a canonical registry. Original records and named scientific review remain necessary for the affected identity, age, acquisition, allocation and reuse decisions. Protocol v1, final splits and ML remain later gates.

## Acceptance

Synthetic tests cover immutable input snapshots, deterministic packet content, preservation of leading zeros and dependent archive copies, repeated references, stale hashes, invalid references/statuses, protocol mismatch, failed extraction, byte budgets, output separation and CLI failure retention. Run the five AGENTS.md checks and `uv run --frozen python scripts/check_source_visibility.py`. The real packet and review brief remain ignored; their paths are indexed in `.local/setup/m03_run_index.json` after validation.
