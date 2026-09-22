"""Seed aliases only. Filename resemblance is never an identity decision."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rat_usv.config import SourceConfig
from rat_usv.provenance.io import record_run, write_json
from rat_usv.schemas.registry import SCHEMAS, CrosswalkEntry


def seed_crosswalk(config: SourceConfig, inventory: Path, out: Path) -> list[dict[str, Any]]:
    rows = []
    for item in pq.read_table(inventory).to_pylist():
        if item["file_type"] != ".wav":
            continue
        rel = Path(item["relative_path"])
        entry = CrosswalkEntry(
            alias=rel.stem,
            namespace=str(rel.parent),
            source_path=str(rel),
            source_locator="filename",
            notes="Observed filename stem only; animal/day/litter meaning unverified. Preserve leading zeros.",
        )
        rows.append(entry.model_dump())
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMAS["identifier_crosswalk"]),
        out / "identifier_crosswalk.parquet",
    )
    write_json(
        out / "summary.json",
        {
            "aliases": len(rows),
            "canonical_assignments": 0,
            "status": "unresolved",
            "evidence_class": "UNRESOLVED",
        },
    )
    (out / "identifier_resolution_report.md").write_text(
        "# Identifier resolution report — candidate ledger\n\nNo biological identities have been assigned. One alias observation per WAV path; repeated stems are kept in separate directory namespaces.\n\nNext: inspect workbook sheets/cells and R transformations; add assertion IDs, exact source hashes and locators; assess date/day semantics, alias reuse, collisions, and conflicting copies. Retain all alternatives and request review before confirmation.\n"
    )
    record_run(out, "crosswalk", config, [inventory])
    return rows
