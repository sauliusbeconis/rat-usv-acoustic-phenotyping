import io
import json
import warnings
import zipfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rat_usv.cli import app
from rat_usv.config import HashPolicy, SourceConfig
from rat_usv.data.inventory import inventory
from rat_usv.evidence.pilot import extract_pilot
from rat_usv.evidence.reconcile import calendar_deltas, reconcile_metadata
from rat_usv.evidence.search import metadata_search
from rat_usv.provenance.io import new_run, sha256_file


def workbook(label):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="r1"/></sheets></workbook>',
        )
        package.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        package.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>{label}</t></is></c></row></sheetData></worksheet>',
        )
    return buffer.getvalue()


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    script = (
        'Diena_90 <- read_excel("prose/g/0001/01.12.xlsx")\n'
        "```{r}\n"
        '# Diena_91 <- read_excel("comment/g/0001/01.12.xlsx")\n'
        'library(readxl)\nDiena_1 <- read_excel("C:/old/g/0001/01.12.xlsx")\n'
        'Diena_2 <- read_excel("C:/old/g/0002/01.13.xlsx")\n'
        'Diena_3 <- read_excel("folder/01.14.xlsx")\n'
        'Diena_4 <- read_excel("g/0001/01.15.xlsx")\n'
        'a <- fct_recode(\n "16" = "01.12",\n "17" = "01.14"\n)\n```\n'
        '```{python}\nDiena_92 <- read_excel("other/g/0001/01.12.xlsx")\n```\n'
    )
    for name in ("first.Rmd", "copy.Rmd"):
        (source / name).write_text(script)
    for tree in ("one", "two"):
        path = source / tree / "g/0001/01.12.xlsx"
        path.parent.mkdir(parents=True)
        path.write_bytes(workbook("same"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(source / "exports.zip", "w") as archive:
            archive.writestr("g/0001/01.12.xlsx", workbook("same"))
            archive.writestr("g/0001/01.12.xlsx", workbook("different"))
    (source / "binary-cache").write_bytes(b"\xff")
    (tmp_path / "protocol.md").write_text("Synthetic protocol")
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "protocol": "protocol.md",
                "sources": [
                    {"path": name, "kind": "script", "selection_reason": "synthetic"}
                    for name in ("first.Rmd", "copy.Rmd")
                ],
            }
        )
    )
    cfg = SourceConfig(
        source_root=source, output_root=tmp_path / "runs", hash=HashPolicy(mode="full")
    )
    evidence = new_run(cfg, "evidence-test")
    assert extract_pilot(cfg, selection, evidence)["success"]
    inv = new_run(cfg, "inventory")
    inventory(cfg, inv)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"patterns": {"session": "Diena"}}))
    search = new_run(cfg, "search-test")
    assert not metadata_search(cfg, inv / "source_inventory.parquet", policy, search)["success"]
    return cfg, evidence, search


def test_reconciliation_preserves_copies_context_and_gaps_without_raw_reads(inputs, monkeypatch):
    cfg, evidence, search = inputs
    before = {
        str(p): sha256_file(p)
        for folder in (evidence, search)
        for p in folder.rglob("*")
        if p.is_file()
    }
    original_open = Path.open

    def no_raw_reads(path, *args, **kwargs):
        assert not path.resolve().is_relative_to(cfg.source_root)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_raw_reads)
    outputs = [new_run(cfg, "reconcile-test") for _ in range(2)]
    for out in outputs:
        summary = reconcile_metadata(cfg, evidence / "evidence.json", search / "entities.json", out)
        assert summary["raw_source_files_opened"] == summary["new_canonical_assignments"] == 0
        assert summary["upstream_search_success"] is False
        assert len(summary["upstream_search_failures"]) == 1
        assert summary["recode_groups"] == 2
        assert summary["non_code_input_examples"] == 4
        assert summary["day_comparison_statuses"] == {"delta_disagreement": 2}
        refs = json.loads((out / "input_references.json").read_text())
        assert len(refs) == 8  # outside prose, comments and Python fences are ignored
        first = refs[0]
        assert first["tail_three_tokens"] == ["g", "0001", "01.12.xlsx"]
        assert first["status"] == "candidates_different_content"
        assert len(first["candidate_entities"]) == 4
        assert first["distinct_candidate_hashes"] == 2
        members = [e for e in first["candidate_entities"] if e["member_name"]]
        assert [e["member_index"] for e in members] == [0, 1]
        assert refs[1]["status"] == refs[3]["status"] == "no_suffix_candidate"
        assert refs[2]["status"] == "insufficient_path_context"
        assert all(r["canonical_animal_id"] is r["recording_session_id"] is None for r in refs)
        assert len(json.loads((out / "mixed_input_contexts.json").read_text())) == 2
    for name in ("input_references.json", "day_recode_groups.json", "summary.json"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    assert all(sha256_file(Path(p)) == digest for p, digest in before.items())


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("12.31", "01.01", [1]),
        ("02.28", "03.01", [1, 2]),
        ("02.29", "03.01", [1]),
        ("01.12", "01.14", [2]),
        ("02.30", "03.01", []),
        ("1.12", "01.14", []),
    ],
)
def test_calendar_hypotheses_preserve_leap_uncertainty(first, second, expected):
    assert calendar_deltas(first, second) == expected


@pytest.mark.parametrize("which", ["evidence", "search"])
def test_stale_derivative_hashes_fail(inputs, which):
    cfg, evidence, search = inputs
    path = evidence / "evidence.json" if which == "evidence" else search / "entities.json"
    with path.open("a") as handle:
        handle.write(" ")
    out = new_run(cfg, "reconcile-test")
    with pytest.raises(ValueError, match="digest mismatch"):
        reconcile_metadata(cfg, evidence / "evidence.json", search / "entities.json", out)
    assert not (out / "input_references.json").exists()


@pytest.mark.parametrize("defect", ["source_hash", "duplicate_entity", "source_verification"])
def test_incompatible_or_invalid_search_is_rejected(inputs, defect):
    cfg, evidence, search = inputs
    name = "summary.json" if defect == "source_verification" else "entities.json"
    payload = json.loads((search / name).read_text())
    if defect == "source_verification":
        payload["source_verification_passed"] = False
    elif defect == "duplicate_entity":
        payload.append(payload[0])
    else:
        next(e for e in payload if e["source_path"] == "first.Rmd")["sha256"] = "0" * 64
    (search / name).write_text(json.dumps(payload))
    run = json.loads((search / "run.json").read_text())
    next(o for o in run["outputs"] if o["path"] == name)["sha256"] = sha256_file(search / name)
    (search / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        reconcile_metadata(
            cfg, evidence / "evidence.json", search / "entities.json", new_run(cfg, "invalid-test")
        )


def test_cli_retains_failure_and_does_not_reuse_output(inputs, tmp_path):
    cfg, evidence, search = inputs
    config = tmp_path / "local.yaml"
    config.write_text("source_root: source\noutput_root: runs\n")
    args = [
        "reconcile-metadata",
        "--config",
        str(config),
        "--evidence",
        str(evidence / "evidence.json"),
        "--search-entities",
        str(search / "entities.json"),
    ]
    ok = CliRunner().invoke(app, args)
    assert ok.exit_code == 0, ok.exception
    with pytest.raises(ValueError, match="new empty"):
        reconcile_metadata(
            cfg, evidence / "evidence.json", search / "entities.json", Path(ok.stdout.strip())
        )
    (search / "entities.json").write_text("[]")
    failed = CliRunner().invoke(app, args)
    assert failed.exit_code == 1
    assert (Path(failed.stdout.strip()) / "fatal_failure.json").is_file()
    assert (Path(failed.stdout.strip()) / "run.json").is_file()
