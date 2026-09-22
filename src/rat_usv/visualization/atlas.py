"""Descriptive technical atlas; biological summaries remain unknown until M0."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from rat_usv.config import SourceConfig
from rat_usv.provenance.io import record_run, write_json


def dataset_atlas(
    config: SourceConfig, inventory: Path, audit: Path | None, out: Path
) -> dict[str, Any]:
    rows = pq.read_table(inventory).to_pylist()
    hashes = Counter(r["sha256"] for r in rows if r.get("sha256"))
    result: dict[str, Any] = {
        "source_files": len(rows),
        "source_bytes": sum(r["size_bytes"] or 0 for r in rows),
        "file_types": dict(Counter(r["file_type"] for r in rows)),
        "inventory_hash_status": dict(Counter(r["hash_status"] for r in rows)),
        "duplicate_hashed_file_groups": sum(v > 1 for v in hashes.values()),
        "biological_counts": {
            "animals": None,
            "litters": None,
            "dams": None,
            "postnatal_day_coverage": None,
            "sex": None,
            "weight": None,
        },
        "limitations": [
            "Filename tokens are not biological identities",
            "Duplicate detection covers only hashed files",
            "No biological counts or developmental conclusions have been inferred",
        ],
    }
    if audit is not None:
        wavs = pq.read_table(audit).to_pylist()
        audio_hashes = Counter(r["sha256"] for r in wavs if r.get("sha256"))
        result["wav"] = {
            "files": len(wavs),
            "total_header_hours": sum(r.get("duration_seconds") or 0 for r in wavs) / 3600,
            "sample_rates": dict(Counter(str(r.get("sample_rate_hz")) for r in wavs)),
            "subtypes": dict(Counter(str(r.get("subtype")) for r in wavs)),
            "channels": dict(Counter(str(r.get("channels")) for r in wavs)),
            "readability": dict(Counter(r["readability"] for r in wavs)),
            "hash_status": dict(Counter(r["hash_status"] for r in wavs)),
            "duplicate_hashed_audio_groups": sum(v > 1 for v in audio_hashes.values()),
        }
    write_json(out / "atlas.json", result)
    (out / "dataset_atlas.md").write_text(
        "# Preliminary technical dataset atlas\n\nThis is an M0 input, not a biological result. Missing biological counts mean unknown, not zero.\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n\nNext audit: reconcile original workbook cells, R transformations, thesis descriptions and recording paths. Check source duplicates, missing sessions, acquisition settings, and candidate litter/day structures before any split design.\n"
    )
    record_run(out, "dataset-atlas", config, [inventory] + ([audit] if audit else []))
    return result
