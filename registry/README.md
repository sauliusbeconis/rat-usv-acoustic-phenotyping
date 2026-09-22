# Local data registry

Run `uv run --frozen python scripts/initialize_registry.py` to create missing valid zero-row Parquet placeholders for source_inventory, animals, litters, dams, recordings, identifier_crosswalk and metadata_assertions, plus empty checksums.sha256. No biological values are invented. Existing files are never overwritten.

All populated/empty parquet and checksum files are ignored. Committed schema definitions are in src/rat_usv/schemas/registry.py and src/rat_usv/data/inventory.py; data_dictionary.md documents semantics. Actual inspection snapshots go to .local/runs, not automatically promoted into the registry. A human-reviewed M0 step will decide promotion and versioning. datasets.yaml describes the source pointer contract without distributing private values.

`usv validate --config configs/data/local.yaml --registry registry` checks structural integrity and reports empty/missing manifests. It always reports m0_ready=false: schema success cannot certify scientific validity. Do not treat absent tables as zero animals. Invalid manifests exit nonzero; pending biological completeness remains explicit warnings.

## Version-2 evidence sidecars

New M0.1 observations and assertions are written to `evidence.json` in new ignored runs. Existing v1 Parquet registries are retained without modification. See `docs/M0_1_EVIDENCE_PILOT.md` for schema semantics and `usv validate-evidence`. The pilot never fills canonical animal/litter/dam registries.
