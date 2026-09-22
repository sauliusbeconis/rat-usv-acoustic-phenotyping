"""Scientific invariants independent of any future training implementation."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import ValidationError

from rat_usv.provenance.io import safe_source, sha256_file
from rat_usv.schemas.registry import SCHEMAS, CrosswalkEntry


def verify_source_hashes(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    failures = []
    for row in rows:
        if not row.get("sha256"):
            continue
        checked += 1
        try:
            path = safe_source(root, row["relative_path"])
            if sha256_file(path) != row["sha256"]:
                failures.append(f"Changed hash: {row['relative_path']}")
        except (OSError, ValueError) as e:
            failures.append(str(e))
    return {
        "checked": checked,
        "unhashed": len(rows) - checked,
        "failures": failures,
        "complete_hash_baseline": checked == len(rows),
    }


def validate_splits(rows: list[dict[str, Any]], litter_holdout: bool = False) -> None:
    """A recording-level manifest must carry biological IDs and audio file hashes."""
    if not rows:
        raise ValueError("An empty split manifest cannot be validated")
    groups = ("animal_id", "audio_sha256") + (("litter_id",) if litter_holdout else ())
    seen: dict[tuple[str, str], str] = {}
    ids: set[str] = set()
    partitions = set()
    for row in rows:
        partition = row.get("split")
        if partition not in {"train", "validation", "test"}:
            raise ValueError("Unknown split label")
        partitions.add(partition)
        rec = row.get("recording_id")
        if not rec or rec in ids:
            raise ValueError("Missing or duplicate recording ID")
        ids.add(rec)
        for key in groups:
            value = row.get(key)
            if not value:
                raise ValueError(f"Missing {key}")
            if key == "audio_sha256" and (
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Invalid audio SHA-256")
            token = (key, value)
            if token in seen and seen[token] != partition:
                raise ValueError(f"{key} overlap across splits: {value}")
            seen[token] = partition
    if not {"train", "test"} <= partitions:
        raise ValueError("Split validation requires train and test")


def validate_fit_scope(fit_log: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> None:
    train = {r["recording_id"] for r in split_rows if r["split"] == "train"}
    if not fit_log:
        raise ValueError("Fit provenance is absent")
    for entry in fit_log:
        ids = set(entry.get("fit_recording_ids", []))
        if not entry.get("component") or not ids or not ids <= train:
            raise ValueError("Fit scope is missing or includes non-training/unknown recordings")


def lock_split(path: Path, rows: list[dict[str, Any]], litter_holdout: bool = False) -> str:
    validate_splits(rows, litter_holdout)
    data = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    with path.open("xb") as handle:
        handle.write(data)
    digest = hashlib.sha256(data).hexdigest()
    # The independently reviewed expected digest must be versioned with the protocol.
    return digest


def verify_locked_split(path: Path, expected_sha256: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Locked split changed")


def validate_registry(directory: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    for name, schema in SCHEMAS.items():
        path = directory / f"{name}.parquet"
        if not path.exists():
            warnings.append(f"Missing manifest: {name}")
            tables[name] = []
            continue
        try:
            table = pq.read_table(path)
            if not table.schema.equals(schema, check_metadata=False):
                errors.append(f"Schema mismatch: {name}")
                tables[name] = []
                continue
            tables[name] = table.to_pylist()
        except (OSError, ValueError) as e:
            errors.append(f"Cannot read {name}: {e}")
            tables[name] = []
    indexes: dict[str, set[str]] = {}
    for name, key in (
        ("animals", "animal_id"),
        ("litters", "litter_id"),
        ("dams", "dam_id"),
        ("recordings", "recording_id"),
        ("metadata_assertions", "assertion_id"),
    ):
        values = [row.get(key) for row in tables[name]]
        if any(not x for x in values) or len(values) != len(set(values)):
            errors.append(f"Missing/duplicate primary key: {name}.{key}")
        indexes[name] = {v for v in values if v}
    for name, column, target in (
        ("animals", "litter_id", "litters"),
        ("animals", "dam_id", "dams"),
        ("litters", "dam_id", "dams"),
        ("recordings", "animal_id", "animals"),
    ):
        for row in tables[name]:
            value = row.get(column)
            if value and value not in indexes[target]:
                errors.append(f"Unknown {name}.{column}: {value}")
            if not value:
                warnings.append(f"Unresolved {name}.{column}")
    for row in tables["recordings"]:
        if row.get("postnatal_day") is not None and row["postnatal_day"] < 0:
            errors.append("Negative postnatal day")
        if row.get("weight_g") is not None and row["weight_g"] <= 0:
            errors.append("Nonpositive weight")
        if row.get("animal_link_status") != "confirmed":
            warnings.append("Recording animal link is not confirmed")
        if row.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            errors.append("Invalid recording SHA-256")
    litter_by_id = {row["litter_id"]: row for row in tables["litters"]}
    for animal in tables["animals"]:
        litter = litter_by_id.get(animal.get("litter_id"))
        if (
            litter
            and animal.get("dam_id")
            and litter.get("dam_id")
            and animal["dam_id"] != litter["dam_id"]
        ):
            errors.append("Animal/litter dam disagreement")
    assertions: dict[tuple[Any, ...], set[str]] = {}
    for row in tables["metadata_assertions"]:
        try:
            json.loads(
                row.get("value_json") or "",
                parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
            )
        except (TypeError, ValueError):
            errors.append("Invalid typed JSON metadata value")
        if row.get("source_sha256") and not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            errors.append("Invalid metadata SHA-256")
        if row.get("evidence_class") not in {
            "EXACT",
            "STRONG",
            "MODERATE",
            "WEAK",
            "CONFLICT",
            "UNRESOLVED",
        }:
            errors.append("Unknown metadata evidence class")
        if row.get("status") not in {
            "confirmed",
            "probable",
            "ambiguous",
            "conflicting",
            "unresolved",
            "superseded",
            "rejected",
        }:
            errors.append("Unknown metadata assertion status")
        if not row.get("source_path") or not row.get("source_locator"):
            errors.append("Metadata assertion lacks source provenance")
        if row.get("evidence_class") == "CONFLICT" and row.get("status") not in {
            "conflicting",
            "superseded",
            "rejected",
        }:
            errors.append("Metadata conflict silently resolved")
        if row.get("status") == "confirmed":
            if not row.get("reviewer") or not row.get("source_sha256"):
                errors.append("Confirmed assertion lacks reviewer/hash")
            assertion_key = (row.get("entity_type"), row.get("entity_id"), row.get("field"))
            assertions.setdefault(assertion_key, set()).add(str(row.get("value_json")))
    if any(len(values) > 1 for values in assertions.values()):
        errors.append("Conflicting confirmed metadata values")
    assertion_by_id = {r.get("assertion_id"): r for r in tables["metadata_assertions"]}
    for row in tables["metadata_assertions"]:
        seen = set()
        current = row
        while current.get("supersedes_assertion_id"):
            target = current["supersedes_assertion_id"]
            if target in seen:
                errors.append("Metadata supersession cycle")
                break
            seen.add(target)
            prior = assertion_by_id.get(target)
            if prior is None:
                errors.append("Unknown superseded assertion")
                break
            if (prior.get("entity_type"), prior.get("entity_id"), prior.get("field")) != (
                row.get("entity_type"),
                row.get("entity_id"),
                row.get("field"),
            ):
                errors.append("Metadata supersession subject mismatch")
            current = prior
    for table_name in ("animals", "litters", "dams", "recordings"):
        for item in tables[table_name]:
            evidence = item.get("evidence_ids") or []
            if not set(evidence) <= indexes["metadata_assertions"]:
                errors.append(f"Unknown evidence ID in {table_name}")
            if not evidence:
                warnings.append(f"No evidence linked in {table_name}")
            entity_type, key = {
                "animals": ("animal", "animal_id"),
                "litters": ("litter", "litter_id"),
                "dams": ("dam", "dam_id"),
                "recordings": ("recording", "recording_id"),
            }[table_name]
            for evidence_id in evidence:
                assertion = assertion_by_id.get(evidence_id)
                if assertion and (assertion.get("entity_type"), assertion.get("entity_id")) != (
                    entity_type,
                    item.get(key),
                ):
                    errors.append(f"Evidence subject mismatch in {table_name}")
    confirmed: dict[tuple[str, str, str], set[str]] = {}
    for row in tables["identifier_crosswalk"]:
        try:
            entry = CrosswalkEntry.model_validate(row)
            if entry.status == "confirmed":
                if not set(entry.evidence_ids) <= indexes["metadata_assertions"]:
                    errors.append("Confirmed mapping references unknown evidence IDs")
                for evidence_id in entry.evidence_ids:
                    assertion = assertion_by_id.get(evidence_id)
                    if assertion and assertion.get("status") != "confirmed":
                        errors.append(
                            "Confirmed mapping relies on unconfirmed/conflicting evidence"
                        )
                    if assertion and (assertion.get("entity_type"), assertion.get("entity_id")) != (
                        entry.entity_type,
                        entry.canonical_id,
                    ):
                        errors.append("Confirmed mapping evidence subject mismatch")
                target_table = {
                    "animal": "animals",
                    "litter": "litters",
                    "dam": "dams",
                    "recording": "recordings",
                }.get(entry.entity_type)
                if target_table and entry.canonical_id not in indexes[target_table]:
                    errors.append("Confirmed mapping references unknown canonical ID")
                confirmed.setdefault((entry.namespace, entry.alias, entry.entity_type), set()).add(
                    str(entry.canonical_id)
                )
            else:
                warnings.append("Unconfirmed crosswalk mapping")
        except ValidationError as e:
            errors.append(str(e))
    if any(len(values) > 1 for values in confirmed.values()):
        errors.append("Alias confirmed to multiple identities in one namespace")
    for name in SCHEMAS:
        if not tables[name]:
            warnings.append(f"Empty manifest: {name}")
    return {
        "schema_valid": not errors,
        "m0_ready": False,
        "m0_note": "Human protocol/provenance review is required; this command never certifies scientific readiness.",
        "rows": {k: len(v) for k, v in tables.items()},
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }
