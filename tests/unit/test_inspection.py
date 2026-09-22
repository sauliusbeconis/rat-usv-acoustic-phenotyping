import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from rat_usv.config import HashPolicy, SourceConfig, load_config
from rat_usv.data.crosswalk import seed_crosswalk
from rat_usv.data.inventory import inventory
from rat_usv.provenance.io import Hasher, new_run, safe_source
from rat_usv.qc.audio import audit, inspect_wav


def test_budget_reports_unhashed_files(tmp_path: Path) -> None:
    f = tmp_path / "large.bin"
    f.write_bytes(b"x" * 20)
    hasher = Hasher(HashPolicy(max_file_bytes=10, max_total_bytes=100))
    assert hasher.compute(f) == (None, "skipped_file_limit")
    hasher = Hasher(HashPolicy(max_file_bytes=30, max_total_bytes=25))
    assert hasher.compute(f)[1] == "hashed"
    assert hasher.compute(f) == (None, "skipped_total_budget")
    assert hasher.bytes_read == 20
    assert Hasher(HashPolicy(mode="none")).compute(f) == (None, "disabled")


def test_full_mode_is_explicit(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_bytes(b"abc")
    assert (
        Hasher(HashPolicy(mode="full", max_file_bytes=0, max_total_bytes=0)).compute(f)[1]
        == "hashed"
    )


def test_prevent_outputs_into_sources(source: Path) -> None:
    with pytest.raises(ValidationError, match="explicit excluded subtree"):
        SourceConfig(source_root=source, output_root=source / "new")
    cfg = SourceConfig(
        source_root=source, output_root=source / "repo" / "runs", exclude_relative=["repo"]
    )
    assert cfg.output_root.is_relative_to(source)


@pytest.mark.parametrize("exclusion", [".", "../escape", "/absolute", ""])
def test_invalid_exclusions(source: Path, tmp_path: Path, exclusion: str) -> None:
    with pytest.raises(ValidationError):
        SourceConfig(
            source_root=source, output_root=tmp_path / "runs", exclude_relative=[exclusion]
        )


def test_relative_config_paths(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml = tmp_path / "local.yaml"
    yaml.write_text("source_root: source\noutput_root: runs\n")
    monkeypatch.chdir("/")
    assert load_config(yaml).source_root == source
    assert load_config(yaml).output_root == tmp_path / "runs"


def test_inventory_excludes_output_and_records_symlinks(source: Path) -> None:
    (source / "linked-file").symlink_to(source / "mapping.csv")
    (source / "linked-dir").symlink_to(source / "session one", target_is_directory=True)
    cfg = SourceConfig(
        source_root=source,
        output_root=source / "repo" / "runs",
        exclude_relative=["repo"],
        hash=HashPolicy(mode="full"),
    )
    out = new_run(cfg, "inventory")
    rows = inventory(cfg, out)
    assert len(rows) == 4
    assert all(not r["relative_path"].startswith("repo/") for r in rows)
    assert (
        next(r for r in rows if r["relative_path"] == "linked-file")["hash_status"]
        == "skipped_nonregular"
    )
    summary = json.loads((out / "summary.json").read_text())
    assert {r["reason"] for r in summary["skipped_subtrees_or_paths"]} == {
        "symlink",
        "configured_exclusion",
    }


def test_safe_source_blocks_escape_and_links(source: Path) -> None:
    (source / "link").symlink_to(source / "mapping.csv")
    for rel in ("../escape", "/etc/passwd", "link"):
        with pytest.raises(ValueError):
            safe_source(source, rel)


def test_wav_header_amplitude_and_channel_separation(source: Path) -> None:
    r = inspect_wav(source / "session one" / "0001.WAV", 1)
    assert r["sample_rate_hz"] == 250000
    assert r["channels"] == 2
    assert r["frames"] == 1000
    assert r["bit_depth"] == 16 and r["sample_width_bytes"] == 2
    assert r["duration_seconds"] == pytest.approx(0.004)
    diagnostics = json.loads(r["diagnostics_json"])
    assert diagnostics["coverage_fraction"] == 1
    channels = diagnostics["windows"][0]["channels"]
    assert channels[0]["full_scale_fraction"] == pytest.approx(0.02)
    assert channels[1]["digital_zero_fraction"] == 1


def test_corrupt_wav_retained(cfg: SourceConfig) -> None:
    (cfg.source_root / "bad.wav").write_bytes(b"not a wave")
    inv = new_run(cfg, "inventory")
    inventory(cfg, inv)
    out = new_run(cfg, "audit")
    rows = audit(cfg, inv / "source_inventory.parquet", out)
    assert len(rows) == 3
    bad = next(r for r in rows if r["relative_path"] == "bad.wav")
    assert bad["readability"] == "unreadable_or_decode_error"
    assert bad["hash_status"] == "hashed"


def test_crosswalk_preserves_leading_zeros_and_context(cfg: SourceConfig) -> None:
    inv = new_run(cfg, "inventory")
    inventory(cfg, inv)
    out = new_run(cfg, "crosswalk")
    rows = seed_crosswalk(cfg, inv / "source_inventory.parquet", out)
    assert [r["alias"] for r in rows] == ["0001", "0001"]
    assert len({r["namespace"] for r in rows}) == 2
    assert all(r["canonical_id"] is None and r["status"] == "unresolved" for r in rows)
    assert pq.read_table(out / "identifier_crosswalk.parquet").num_rows == 2


def test_empty_source_has_typed_table(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    cfg = SourceConfig(source_root=root, output_root=tmp_path / "out")
    out = new_run(cfg, "inventory")
    inventory(cfg, out)
    assert (
        pq.read_table(out / "source_inventory.parquet").schema.field("size_bytes").type.bit_width
        == 64
    )
