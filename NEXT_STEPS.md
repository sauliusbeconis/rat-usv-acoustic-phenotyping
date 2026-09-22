# Next step: original records and scientific review

M0.4 adds bounded discovery and extraction of marking, birth and recording-session evidence from an explicitly inventoried metadata root. See `docs/M0_4_METADATA_DISCOVERY.md` and decision 0006. After validation, `.local/setup/m04_run_index.json` indexes the private extraction and review handoff. Historical descriptions and code recodes remain unconfirmed; original individual logs and source-tree correspondence require separate evidence.

M0.3 prepares a traceable review packet from the existing M0.1 derivatives. See `docs/M0_3_REVIEW_PACKET.md` and decision 0004. It preserves every open ledger item, expands original evidence references and explicitly identifies missing links. After validation, `.local/setup/m03_run_index.json` indexes the private packet and source-specific review brief. This is review preparation; it does not resolve scientific decisions or reopen raw sources.

The M0 baseline is delivered in Git. Publication includes implementation, generic documentation, locked dependencies and synthetic tests only. Source-specific layout and inventory findings remain in private handoffs. Committing or pushing infrastructure does not advance the scientific gates below.

M0.2 adds full-decode technical QC for the same four WAVs exposed in M0.1. See `docs/M0_2_FULL_DECODE_PILOT.md` and decision 0003 for its bounded scope and reporting cutoffs. After validation, `.local/setup/m02_run_index.json` indexes the private handoff. Decoding coverage and digital artifacts do not determine biological validity or exclusion.

The approved bounded M0.1 implementation adds source observations, contextual historical assertions, candidate relationships and review validation. See `docs/M0_1_EVIDENCE_PILOT.md` for commands, schema migration, parser limits and acceptance checks. The real pilot outputs remain in new ignored `.local/runs/` directories; the latest handoff is indexed by `.local/setup/m01_run_index.json` after validation.

Next, review the private conflict ledger and original records to resolve identity continuity, anomalous folders, day/age conventions, acquisition units and reuse rights. Canonical biological assignments remain empty. Full source freeze/backup, collection-wide decoding, annotation, protocol v1, final splits and ML remain later gates. The M0.2 continuation authorizes only the explicitly bounded technical decode pilot.

## Earlier scaffold handoff (historical)

The repository infrastructure is ready for review; the biological dataset is not reconstructed. Start by reading `.local/setup/SETUP_REPORT.md` and `.local/setup/run_index.json` for actual findings and run paths. Registry placeholders are empty; candidate alias rows are unresolved. Hashing coverage is limited and explicit.

## Continue with the installed Codex CLI

```bash
cd /path/to/rat-usv-acoustic-phenotyping
uv sync --frozen --group dev --group workflow
uv run --frozen pytest
codex --sandbox read-only
```

Inside the interactive CLI, enter:

```text
/plan Read docs/M0_AUDIT_PROMPT.md and follow it. Audit the actual sources and propose M0 only; do not change files or train models.
```

The local executable is `codex` (0.155.1 checked during setup), not a separate `gpt-cli`. `codex --sandbox read-only` is supported by local help; `/plan` is documented in the [official CLI commands](https://developers.openai.com/codex/cli/slash-commands). Do not pass `/plan` as a shell command. Read-only mode protects the planning phase; a later explicitly approved M0 implementation session can use workspace-write from this repository. Do not add the source parent as a writable directory for routine research.

## Proposed audit order

1. Review inventory/audit errors, hash coverage, duplicate historical trees and exact meanings of directory tokens. Establish a full source freeze/backup plan; avoid unnecessary repeated full-collection reads.
2. Inspect a representative workbook from each format and the R/Rmd transformation code. Record exact sheet/cell/formula and script/line locators; relate thesis definitions cautiously. Determine whether Avisoft exports are direct acoustic measurements and document columns/units.
3. Build atomic metadata assertions and candidate mappings with EXACT/STRONG/MODERATE/WEAK/CONFLICT/UNRESOLVED evidence classes. Preserve ambiguous/conflicting aliases and both source copies. Request specific evidence for unresolved links.
4. Review animal/litter/dam identities, day chronology and weight/sex metadata before promoting into canonical registries. The source filename stem alone is insufficient.
5. Complete recording/QC registry and a biological Dataset Atlas with missingness and conflict denominators. Diagnose clipping/detection over full recordings only under an explicit bounded plan.
6. Propose protocol v1, independent sample-size feasibility, baseline endpoint and protected split design. No final split or ML until this review is complete.

Human scientific decisions: primary endpoint/estimand; unresolved identity conflicts; acquisition interpretation; P0/P1 convention; ethics/data rights; exclusion thresholds; biological vs technical identifiability; realistic external validation cohort. These are future M0 decisions, not blockers to reviewing the delivered scaffold.
