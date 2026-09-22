# Data dictionary v0

All biological IDs/aliases are strings. Preserve leading zeros and exact source spelling. Null means unknown; never use 0, empty string, "NA" or a guessed value as a missing ID. Paths are source-relative, with their configured root and source manifest hash recorded by the run. Dates require source convention and timezone resolution; birth_date is currently a string pending a reviewed format.

| Table | Key / core fields | Semantics and constraints |
|---|---|---|
| source_inventory | relative_path; file_type, size_bytes, mtime_ns, sha256, hash_status, role, parser_status, notes | One filesystem entry, bytes/ns units; extension-based role is provisional; SHA null when uncomputed; errors/skips explicit |
| animals | animal_id; litter_id, dam_id, sex, evidence_ids | Canonical animal; links reference known entities if populated; sex vocabulary must be defined from evidence |
| litters | litter_id; dam_id, birth_date, evidence_ids | Litter hierarchy; no inferred dam identity |
| dams | dam_id, evidence_ids | Maternal identity with source evidence |
| recordings | recording_id; relative_path, animal_id, animal_link_status, postnatal_day, weight_g, session_id, sha256, evidence_ids | Unknown animal link retained as null; day integer >=0 after P0/P1 reconciliation; weight in grams >0 when evidenced and linked to relevant time |
| identifier_crosswalk | alias + namespace + source_path + source_locator; entity_type, canonical_id, status, evidence_class, evidence_ids, reviewer, reviewed_utc, notes | Observation ledger, not a unique-alias collapse; confirmed requires reviewed exact/strong evidence; conflicting alternatives retained |
| metadata_assertions | assertion_id; entity_type/id, field, value_json, source_path/hash/locator, evidence_class, status, supersedes_assertion_id, reviewer, notes | Original atomic source assertion; JSON-encoded typed value with units in field schema/notes; retain old assertions; explicit supersession/rejection |
| WAV audit | relative_path, size_bytes, sample_rate_hz, channels, subtype, bit_depth, sample_width_bytes, frames, duration_seconds, readability, sha256, hash_status, diagnostics_json, notes | Header duration; valid bit depth unknown if subtype unsupported; bounded per-channel amplitude diagnostics, window offsets and coverage |

Assertion statuses add `superseded` and `rejected` to mapping statuses so obsolete statements stay traceable. Confirmed contradictory values cannot silently coexist. Alias namespaces must be narrowed to session/litter/date context where required by evidence; automatic command uses the observed parent directory only and makes no biological assignment.

Future split JSON rows require recording_id, animal_id, audio_sha256, split (`train`, `validation`, `test`), and litter_id for litter holdout. Hash must be real 64-character lowercase SHA-256. Fit-log JSON entries require component and fit_recording_ids. Fit components include scaler, imputer, feature selection, dimensionality reduction, encoder and any data-driven preprocessing. The ledger is a declared provenance check; integration into each future training pipeline remains required.

WAV full_scale_fraction is the fraction of sampled finite values at digital PCM endpoints (or float |x|>=1); it is only a clipping proxy. digital_zero_fraction is exact zero, not an acoustic silence classifier. Decoder errors remain rows. Sampled decode success does not certify every frame.
