import json
import zipfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rat_usv.cli import app
from rat_usv.config import SourceConfig
from rat_usv.evidence import review
from rat_usv.evidence.pilot import extract_pilot
from rat_usv.provenance.io import new_run, sha256_file


@pytest.fixture
def extraction(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    script = (
        "```{r}\npateles_1_vada <- filter(x, vados_nr == 1 & individo_id %in% c(0001))\n"
        'a <- fct_recode(\n "16" = "01.12",\n "17" = "01.14"\n)\n```\n'
    )
    (source / "first.Rmd").write_text(script)
    (source / "copy.Rmd").write_text(script)
    with zipfile.ZipFile(source / "history.zip", "w") as archive:
        archive.writestr("copy.Rmd", script)
    (tmp_path / "protocol.md").write_text("Synthetic draft protocol")
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "protocol": "protocol.md",
                "sources": [
                    {"path": name, "kind": "script", "selection_reason": "synthetic"}
                    for name in ("first.Rmd", "copy.Rmd")
                ]
                + [
                    {
                        "path": "history.zip",
                        "kind": "archive",
                        "members": ["copy.Rmd"],
                        "selection_reason": "synthetic dependent copy",
                    }
                ],
                "unresolved_decisions": ["Obtain original marking logs. ```Untrusted text```"],
            }
        )
    )
    cfg = SourceConfig(source_root=source, output_root=tmp_path / "runs")
    out = new_run(cfg, "synthetic-extraction")
    assert extract_pilot(cfg, selection, out)["success"]
    return cfg, out


def replace_recorded_input(root, name, value):
    """Update only synthetic test derivatives and their local audit record."""
    (root / name).write_text(json.dumps(value))
    run = json.loads((root / "run.json").read_text())
    for row in run["outputs"]:
        if row["path"] == name:
            row["sha256"] = sha256_file(root / name)
    (root / "run.json").write_text(json.dumps(run))


def test_review_preserves_evidence_without_opening_raw_sources(extraction, monkeypatch):
    cfg, upstream = extraction
    before = {p.name: sha256_file(p) for p in upstream.iterdir() if p.is_file()}
    original = json.loads((upstream / "evidence.json").read_text())
    original_ledger = json.loads((upstream / "conflict_ledger.json").read_text())
    original_open = Path.open

    def no_source_read(path, *args, **kwargs):
        assert not path.resolve().is_relative_to(cfg.source_root), "Raw source opened"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_source_read)
    outputs = [new_run(cfg, "review-test") for _ in range(2)]
    for out in outputs:
        summary = review.review_packet(cfg, upstream / "evidence.json", out)
        assert summary["items_without_evidence_links"] == 1
        assert summary["new_review_decisions"] == summary["new_canonical_assignments"] == 0
        assert summary["scientific_readiness_assessed"] is False
        packet = json.loads((out / "review_packet.json").read_text())
        assert [i["ledger_entry"] for i in packet] == original_ledger
        expanded = json.loads((out / "assertions.json").read_text())
        assert [a["assertion"] for a in expanded] == original["assertions"]
        assert all(a["assertion"]["canonical_id"] is None for a in expanded)
        aliases = [a for a in expanded if a["assertion"]["field"] == "sex_recode"]
        assert len(aliases) == 3  # loose copies and archive member all remain
        assert all(a["assertion"]["context"]["alias"] == "0001" for a in aliases)
        assert aliases[-1]["evidence"][0]["artifact"]["member_name"] == "copy.Rmd"
        assert aliases[-1]["evidence"][0]["artifact"]["member_index"] == 0
        relationships = json.loads((out / "relationships.json").read_text())
        assert [e["relationship"] for e in relationships] == original["relationships"]
        for name in review.INPUT_NAMES:
            assert sha256_file(out / "inputs" / name) == before[name]
        run = json.loads((out / "run.json").read_text())
        for row in run["outputs"]:
            assert sha256_file(out / row["path"]) == row["sha256"]
    for name in ("review_packet.json", "assertions.json", "relationships.json", "review.md"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    assert {p.name: sha256_file(p) for p in upstream.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("name", ["evidence.json", "conflict_ledger.json", "protocol_snapshot.md"])
def test_stale_upstream_digests_fail_before_review(extraction, name):
    cfg, upstream = extraction
    with (upstream / name).open("a") as handle:
        handle.write(" ")
    out = new_run(cfg, "review-test")
    with pytest.raises(ValueError, match="Upstream output digest mismatch"):
        review.review_packet(cfg, upstream / "evidence.json", out)
    assert not (out / "review_packet.json").exists()


@pytest.mark.parametrize("defect", ["assertion", "observation", "duplicate", "status"])
def test_ledger_validation(extraction, defect):
    cfg, upstream = extraction
    ledger = json.loads((upstream / "conflict_ledger.json").read_text())
    if defect == "duplicate":
        ledger.append(ledger[0])
    elif defect == "status":
        ledger[0]["status"] = "CONFIRMED"
    else:
        ledger[0][defect + "_ids"] = ["does-not-exist"]
    replace_recorded_input(upstream, "conflict_ledger.json", ledger)
    with pytest.raises(ValueError):
        review.review_packet(cfg, upstream / "evidence.json", new_run(cfg, "review-test"))


def test_repeated_observation_references_are_preserved(extraction):
    cfg, upstream = extraction
    ledger = json.loads((upstream / "conflict_ledger.json").read_text())
    ledger[1]["observation_ids"] *= 2
    replace_recorded_input(upstream, "conflict_ledger.json", ledger)
    out = new_run(cfg, "review-test")
    review.review_packet(cfg, upstream / "evidence.json", out)
    packet = json.loads((out / "review_packet.json").read_text())
    assert len(packet[1]["observations"]) == 2
    assert packet[1]["observations"][0] == packet[1]["observations"][1]


@pytest.mark.parametrize("defect", ["failed_run", "protocol", "budget", "symlink", "output"])
def test_review_prerequisites(extraction, monkeypatch, defect):
    cfg, upstream = extraction
    out = new_run(cfg, "review-test")
    if defect == "failed_run":
        replace_recorded_input(upstream, "failures.json", [{"error": "synthetic failure"}])
    elif defect == "protocol":
        replace_recorded_input(upstream, "protocol_snapshot.md", "different protocol")
    elif defect == "budget":
        monkeypatch.setattr(review, "MAX_TOTAL_BYTES", 5)
    elif defect == "symlink":
        ledger = upstream / "conflict_ledger.json"
        ledger.rename(upstream / "ledger-backup.json")
        ledger.symlink_to(upstream / "ledger-backup.json")
    else:
        out = upstream
    with pytest.raises(ValueError):
        review.review_packet(cfg, upstream / "evidence.json", out)


def test_cli_success_and_failure_retention(extraction, tmp_path):
    cfg, upstream = extraction
    config_path = tmp_path / "local.yaml"
    config_path.write_text("source_root: source\noutput_root: runs\n")
    args = [
        "review-packet",
        "--config",
        str(config_path),
        "--evidence",
        str(upstream / "evidence.json"),
    ]
    success = CliRunner().invoke(app, args)
    assert success.exit_code == 0, success.exception
    assert (Path(success.stdout.strip()) / "review.md").is_file()
    (upstream / "conflict_ledger.json").write_text("[]")
    failed = CliRunner().invoke(app, args)
    assert failed.exit_code == 1
    failure_path = Path(failed.stdout.strip())
    assert failure_path.is_relative_to(cfg.output_root)
    assert "digest mismatch" in (failure_path / "fatal_failure.json").read_text()
    assert json.loads((failure_path / "run.json").read_text())["command"] == "review-packet-failed"
    assert not (failure_path / "review.md").exists()


def test_source_markdown_cannot_close_json_fence():
    text = review.fenced("before ```json after ````")
    assert text.startswith("`````json\n")
    assert text.endswith("\n`````\n")
