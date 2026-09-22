# 0002 — M0.1 observations before biological assignment

Date: 2026-09-22. Status: implemented infrastructure; scientific review pending.

The implementation request authorizes the bounded M0.1 evidence pilot, not later M0 completion. Add a strict version-2 JSON evidence sidecar in new private runs while retaining the existing v1 Parquet registries. Preserve source bytes, original cell/metadata representations, conflicting assertions, aliases and dependent copies. Candidate relationships retain alternatives; no canonical biological assignments are generated.

Full before/after hashes cover only explicit pilot inputs. Metadata parsing skips WAV audio payloads; hashing does read them. PDF text and lexical R interpretation remain historical evidence with limited scope. Review and manifest validation are audit controls, not tamper-proof storage or verification of the underlying biology.

Repair Git visibility with narrow Python-package and example-config exceptions. All private-data exclusions remain. The existing untracked scaffold is retained without staging, committing or pushing. Validation uses synthetic fixtures and an isolated candidate checkout. Usage, schema semantics and parser limits are in `docs/M0_1_EVIDENCE_PILOT.md`.
