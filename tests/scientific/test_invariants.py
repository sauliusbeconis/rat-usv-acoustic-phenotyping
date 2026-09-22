from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from rat_usv.config import SourceConfig
from rat_usv.data.inventory import inventory
from rat_usv.provenance.io import new_run, sha256_file
from rat_usv.provenance.validation import (
    lock_split,
    validate_fit_scope,
    validate_registry,
    validate_splits,
    verify_locked_split,
    verify_source_hashes,
)
from rat_usv.qc.audio import audit
from rat_usv.schemas.registry import SCHEMAS, CrosswalkEntry, initialize_empty

pytestmark = pytest.mark.scientific


def test_valid_animal_and_litter_holdout(split_rows: list[dict[str, str]]) -> None:
    validate_splits(split_rows, litter_holdout=True)


@pytest.mark.parametrize("key", ["animal_id", "litter_id", "audio_sha256"])
def test_no_biological_or_duplicate_audio_overlap(
    split_rows: list[dict[str, str]], key: str
) -> None:
    split_rows[-1][key] = split_rows[0][key]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(split_rows, litter_holdout=True)


@pytest.mark.parametrize(
    "component", ["scaler", "imputer", "feature_selection", "dimred", "ssl_encoder"]
)
def test_test_set_never_in_fit_scope(split_rows: list[dict[str, str]], component: str) -> None:
    with pytest.raises(ValueError, match="non-training"):
        validate_fit_scope(
            [{"component": component, "fit_recording_ids": ["r1", "r3"]}], split_rows
        )
    validate_fit_scope([{"component": component, "fit_recording_ids": ["r1"]}], split_rows)


def test_validation_and_unknown_ids_cannot_fit(split_rows: list[dict[str, str]]) -> None:
    for ids in (["r2"], ["unknown"], []):
        with pytest.raises(ValueError):
            validate_fit_scope([{"component": "scaler", "fit_recording_ids": ids}], split_rows)


def test_locked_split_is_write_once_and_tampering_detected(
    tmp_path: Path, split_rows: list[dict[str, str]]
) -> None:
    path = tmp_path / "split.json"
    digest = lock_split(path, split_rows, litter_holdout=True)
    verify_locked_split(path, digest)
    with pytest.raises(FileExistsError):
        lock_split(path, split_rows)
    path.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        verify_locked_split(path, digest)


def test_raw_hashes_unchanged_by_inventory_and_audit(cfg: SourceConfig) -> None:
    before = {p: sha256_file(p) for p in cfg.source_root.rglob("*") if p.is_file()}
    inv = new_run(cfg, "inventory")
    rows = inventory(cfg, inv)
    audit(cfg, inv / "source_inventory.parquet", new_run(cfg, "audit"))
    assert before == {p: sha256_file(p) for p in before}
    assert verify_source_hashes(cfg.source_root, rows)["failures"] == []
    (cfg.source_root / "mapping.csv").write_text("changed")
    assert verify_source_hashes(cfg.source_root, rows)["failures"]


def test_unknown_animal_reference_fails(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    rows = [
        {
            "recording_id": "r1",
            "relative_path": "recording.wav",
            "animal_id": "invented",
            "animal_link_status": "confirmed",
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMAS["recordings"]), tmp_path / "recordings.parquet"
    )
    assert "Unknown recordings.animal_id: invented" in validate_registry(tmp_path)["errors"]


def test_conflicting_metadata_cannot_silently_be_confirmed(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    rows = [
        {
            "assertion_id": "e1",
            "entity_type": "animal",
            "entity_id": "a1",
            "field": "sex",
            "value_json": '"female"',
            "source_path": "historical.xlsx",
            "source_locator": "Sheet1!A2",
            "evidence_class": "CONFLICT",
            "status": "confirmed",
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMAS["metadata_assertions"]),
        tmp_path / "metadata_assertions.parquet",
    )
    assert "Metadata conflict silently resolved" in validate_registry(tmp_path)["errors"]


def test_contradictory_confirmed_values_fail(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    rows = [
        {
            "assertion_id": f"e{i}",
            "entity_type": "animal",
            "entity_id": "a1",
            "field": "sex",
            "value_json": value,
            "source_path": "evidence.csv",
            "source_sha256": "a" * 64,
            "source_locator": f"row {i}",
            "evidence_class": "EXACT",
            "status": "confirmed",
            "reviewer": "synthetic reviewer",
        }
        for i, value in enumerate(['"female"', '"male"'])
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMAS["metadata_assertions"]),
        tmp_path / "metadata_assertions.parquet",
    )
    assert "Conflicting confirmed metadata values" in validate_registry(tmp_path)["errors"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "confirmed", "canonical_id": "a1"},
        {"evidence_class": "CONFLICT"},
        {"canonical_id": "a1"},
    ],
)
def test_mapping_promotion_requires_review(overrides: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        CrosswalkEntry.model_validate(
            {
                "alias": "0001",
                "namespace": "n",
                "source_path": "p",
                "source_locator": "filename",
                **overrides,
            }
        )


def test_empty_registry_never_means_scientific_readiness(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    report = validate_registry(tmp_path)
    assert report["schema_valid"]
    assert report["m0_ready"] is False
    assert report["warnings"]


def test_initializer_does_not_overwrite(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    p = tmp_path / "animals.parquet"
    before = p.stat().st_mtime_ns
    initialize_empty(tmp_path)
    assert p.stat().st_mtime_ns == before


def test_confirmed_mapping_cannot_rely_on_conflicting_assertion(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    evidence = [
        {
            "assertion_id": "e1",
            "entity_type": "animal",
            "entity_id": "a1",
            "field": "identity",
            "value_json": '"a1"',
            "source_path": "e.csv",
            "source_locator": "row 1",
            "evidence_class": "CONFLICT",
            "status": "conflicting",
        }
    ]
    mapping = CrosswalkEntry(
        alias="0001",
        namespace="test",
        source_path="e.csv",
        source_locator="row 1",
        canonical_id="a1",
        entity_type="animal",
        status="confirmed",
        evidence_class="EXACT",
        evidence_ids=["e1"],
        reviewer="synthetic",
        reviewed_utc="2026-01-01T00:00:00Z",
    )
    pq.write_table(
        pa.Table.from_pylist(evidence, schema=SCHEMAS["metadata_assertions"]),
        tmp_path / "metadata_assertions.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([mapping.model_dump()], schema=SCHEMAS["identifier_crosswalk"]),
        tmp_path / "identifier_crosswalk.parquet",
    )
    assert (
        "Confirmed mapping relies on unconfirmed/conflicting evidence"
        in validate_registry(tmp_path)["errors"]
    )


def test_hierarchy_dam_conflict_and_evidence_subject(tmp_path: Path) -> None:
    initialize_empty(tmp_path)
    rows = {
        "dams": [{"dam_id": "d1"}, {"dam_id": "d2"}],
        "litters": [{"litter_id": "l1", "dam_id": "d1"}],
        "animals": [{"animal_id": "a1", "litter_id": "l1", "dam_id": "d2", "evidence_ids": ["e1"]}],
        "metadata_assertions": [
            {
                "assertion_id": "e1",
                "entity_type": "animal",
                "entity_id": "other",
                "field": "sex",
                "value_json": '"female"',
                "source_path": "synthetic",
                "source_locator": "line:1",
                "evidence_class": "MODERATE",
                "status": "unresolved",
            }
        ],
    }
    for name, values in rows.items():
        pq.write_table(
            pa.Table.from_pylist(values, schema=SCHEMAS[name]), tmp_path / f"{name}.parquet"
        )
    errors = validate_registry(tmp_path)["errors"]
    assert "Animal/litter dam disagreement" in errors
    assert "Evidence subject mismatch in animals" in errors


@pytest.mark.parametrize("value", ["not JSON", "NaN", "Infinity", '{"bad":}'])
def test_metadata_json_hash_and_supersession_validation(tmp_path: Path, value: str) -> None:
    initialize_empty(tmp_path)
    rows = [
        {
            "assertion_id": f"e{i}",
            "entity_type": "animal",
            "entity_id": "a1",
            "field": "sex",
            "value_json": value,
            "source_path": "synthetic",
            "source_locator": f"line:{i}",
            "source_sha256": "invalid",
            "evidence_class": "MODERATE",
            "status": "superseded",
            "supersedes_assertion_id": f"e{1 - i}",
        }
        for i in range(2)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMAS["metadata_assertions"]),
        tmp_path / "metadata_assertions.parquet",
    )
    errors = validate_registry(tmp_path)["errors"]
    assert "Invalid typed JSON metadata value" in errors
    assert "Invalid metadata SHA-256" in errors
    assert "Metadata supersession cycle" in errors
