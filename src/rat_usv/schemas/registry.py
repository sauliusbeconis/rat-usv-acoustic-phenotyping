"""Version-one empty registry schemas. Null is unknown, never a guessed value."""

from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

MappingStatus = Literal["confirmed", "probable", "ambiguous", "conflicting", "unresolved"]
EvidenceClass = Literal["EXACT", "STRONG", "MODERATE", "WEAK", "CONFLICT", "UNRESOLVED"]


class CrosswalkEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    entity_type: Literal["animal", "litter", "dam", "recording", "day", "unknown"] = "unknown"
    canonical_id: str | None = None
    status: MappingStatus = "unresolved"
    evidence_class: EvidenceClass = "UNRESOLVED"
    evidence_ids: list[str] = Field(default_factory=list)
    reviewer: str | None = None
    reviewed_utc: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def evidence_required(self) -> "CrosswalkEntry":
        if self.status == "confirmed" and (
            not self.canonical_id
            or not self.reviewer
            or not self.reviewed_utc
            or not self.evidence_ids
            or self.evidence_class not in {"EXACT", "STRONG"}
        ):
            raise ValueError(
                "Confirmed mappings require canonical ID, reviewer, time and exact/strong evidence"
            )
        if self.evidence_class == "CONFLICT" and self.status != "conflicting":
            raise ValueError("Conflicting evidence cannot be silently resolved")
        if self.status == "unresolved" and self.canonical_id is not None:
            raise ValueError("Unresolved mappings cannot assign a canonical identity")
        return self


S = pa.string()
SCHEMAS: dict[str, Any] = {
    "animals": pa.schema(
        [
            ("animal_id", S),
            ("litter_id", S),
            ("dam_id", S),
            ("sex", S),
            ("evidence_ids", pa.list_(S)),
        ]
    ),
    "litters": pa.schema(
        [("litter_id", S), ("dam_id", S), ("birth_date", S), ("evidence_ids", pa.list_(S))]
    ),
    "dams": pa.schema([("dam_id", S), ("evidence_ids", pa.list_(S))]),
    "recordings": pa.schema(
        [
            ("recording_id", S),
            ("relative_path", S),
            ("animal_id", S),
            ("animal_link_status", S),
            ("postnatal_day", pa.int64()),
            ("weight_g", pa.float64()),
            ("session_id", S),
            ("sha256", S),
            ("evidence_ids", pa.list_(S)),
        ]
    ),
    "identifier_crosswalk": pa.schema(
        [
            ("alias", S),
            ("namespace", S),
            ("source_path", S),
            ("source_locator", S),
            ("entity_type", S),
            ("canonical_id", S),
            ("status", S),
            ("evidence_class", S),
            ("evidence_ids", pa.list_(S)),
            ("reviewer", S),
            ("reviewed_utc", S),
            ("notes", S),
        ]
    ),
    "metadata_assertions": pa.schema(
        [
            ("assertion_id", S),
            ("entity_type", S),
            ("entity_id", S),
            ("field", S),
            ("value_json", S),
            ("source_path", S),
            ("source_sha256", S),
            ("source_locator", S),
            ("evidence_class", S),
            ("status", S),
            ("supersedes_assertion_id", S),
            ("reviewer", S),
            ("notes", S),
        ]
    ),
}


def initialize_empty(directory: Path) -> None:
    """Create missing valid zero-row parquet files, never overwrite a manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, schema in SCHEMAS.items():
        target = directory / f"{name}.parquet"
        if not target.exists():
            with target.open("xb") as stream:
                pq.write_table(pa.Table.from_pylist([], schema=schema), stream)
