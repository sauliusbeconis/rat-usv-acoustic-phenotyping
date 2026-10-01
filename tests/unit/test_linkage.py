import csv
import json

import pytest
from typer.testing import CliRunner

from rat_usv.cli import app
from rat_usv.data.linkage import check_sources, final_pup_recordings, load_recordings, summarize

FIELDS = [
    "wav_path", "recording_date", "date_folder", "group_folder", "animal_id", "marking", "sex",
    "litter_id", "dam_id", "birth_date", "postnatal_age_days", "weight_g", "weight_scan",
    "weight_scan_row", "in_final_record_list", "exclusion_reason", "wav_duration_s",
    "edit_suspected", "wav_size_bytes", "identity_basis",
]  # fmt: skip


def pup(path="0102/1_2/0001.wav", animal="E9L1-0001", marking="0.0.0.1", day="2020-01-02", age=1):
    return {
        "wav_path": path, "recording_date": day, "date_folder": path[:4], "group_folder": "1_2",
        "animal_id": animal, "marking": marking, "sex": "female", "litter_id": "E9L1",
        "dam_id": "E9D1", "birth_date": "2020-01-01", "postnatal_age_days": str(age),
        "weight_g": "6.5", "weight_scan": "scan.jpg", "weight_scan_row": "scan-r1",
        "in_final_record_list": "True", "exclusion_reason": "", "wav_duration_s": "301.0",
        "edit_suspected": "False", "wav_size_bytes": "4", "identity_basis": "synthetic",
    }  # fmt: skip


def ambient():
    row = {k: "" for k in FIELDS}
    row.update(
        wav_path="0102/noise/room.wav", date_folder="0102", group_folder="noise",
        in_final_record_list="True", exclusion_reason="not a pup recording",
        wav_duration_s="301.0", edit_suspected="False", wav_size_bytes="4", identity_basis="none",
    )  # fmt: skip
    return row


def write(tmp_path, rows):
    with (tmp_path / "recordings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_valid_linkage_summary(tmp_path):
    rows = load_recordings(write(tmp_path, [pup(), ambient()]))
    assert [r.animal_id for r in final_pup_recordings(rows)] == ["E9L1-0001"]
    summary = summarize(rows)
    assert summary["animals"] == 1 and summary["final_pup_recordings"] == 1
    assert summary["age_coverage"] == {"E9L1": {"min": 1, "max": 1, "missing": []}}


@pytest.mark.parametrize(
    "row, message",
    [
        (pup(animal="E9L1-1000"), "does not encode marking"),
        (pup(marking="0001"), "X.X.X.X"),
        (pup(age=2), "P0"),
        (pup(path="../escape.wav"), "relative"),
    ],
)
def test_rejects_inconsistent_rows(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        load_recordings(write(tmp_path, [row]))


def test_rejects_two_final_recordings_per_animal_day(tmp_path):
    rows = [pup(), pup(path="0102/1_2/0001k.wav")]
    with pytest.raises(ValueError, match="animal-day"):
        load_recordings(write(tmp_path, rows))


def test_check_sources_reports_missing_and_changed(tmp_path):
    source = tmp_path / "source"
    (source / "0102/1_2").mkdir(parents=True)
    (source / "0102/1_2/0001.wav").write_bytes(b"12345")
    rows = load_recordings(write(tmp_path, [pup(), ambient()]))
    problems = {p["wav_path"]: p["problem"] for p in check_sources(rows, source)}
    assert problems == {"0102/1_2/0001.wav": "size_changed", "0102/noise/room.wav": "missing"}


def test_linkage_check_cli(tmp_path):
    source = tmp_path / "source"
    (source / "0102/1_2").mkdir(parents=True)
    (source / "0102/1_2/0001.wav").write_bytes(b"1234")
    (tmp_path / "linkage").mkdir()
    linkage = write(tmp_path / "linkage", [pup()])
    config = tmp_path / "config.yaml"
    config.write_text(f"source_root: source\noutput_root: runs\nlinkage_dir: {linkage}\n")
    result = CliRunner().invoke(app, ["linkage-check", "--config", str(config)])
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").iterdir())
    report = json.loads((run / "linkage_check.json").read_text())
    assert report["source_problems"] == [] and report["animals"] == 1
    assert (run / "run.json").exists()
