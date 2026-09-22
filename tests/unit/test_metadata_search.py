import io
import json
import subprocess
import warnings
import zipfile

import pytest

from rat_usv.config import HashPolicy, SourceConfig
from rat_usv.data.inventory import inventory
from rat_usv.evidence.search import metadata_search, text_records
from rat_usv.provenance.io import new_run, sha256_file


def workbook_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as package:
        package.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Log" sheetId="1" r:id="r1"/></sheets></workbook>',
        )
        package.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        package.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>birth record 0001</t></is></c><c r="B1"><f>birth_date</f><v>0</v></c></row></sheetData></worksheet>',
        )
    return stream.getvalue()


@pytest.fixture
def search_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.R").write_text('birth <- "unknown"\nstop("must never execute")\n')
    (source / "fixture.xlsx").write_bytes(workbook_bytes())
    (source / ".RData").write_bytes(b"binary birth must not deserialize")
    (source / "report.html").write_text("<script>birth hidden</script><p>birth visible</p>")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(source / "history.zip", "w") as archive:
            archive.writestr("log.txt", "birth alias 0001")
            archive.writestr("log.txt", "birth alias 0011u")
            archive.writestr("../log.txt", "birth unsafe")
    cfg = SourceConfig(
        source_root=source, output_root=tmp_path / "runs", hash=HashPolicy(mode="full")
    )
    inv = new_run(cfg, "inventory")
    inventory(cfg, inv)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"patterns": {"birth": "birth"}}))
    return cfg, inv / "source_inventory.parquet", policy


def test_metadata_search_preserves_sources_and_original_cells(search_input):
    cfg, inv, policy = search_input
    before = {p.name: sha256_file(p) for p in cfg.source_root.iterdir()}
    out = new_run(cfg, "search")
    result = metadata_search(cfg, inv, policy, out)
    assert result["success"] and result["source_verification_passed"]
    assert result["canonical_assignments"] == 0
    entities = json.loads((out / "entities.json").read_text())
    matches = [json.loads(line) for line in (out / "matches.jsonl").read_text().splitlines()]
    assert any(e["status"] == "not_deserialized" for e in entities)
    assert any(e["status"] == "unsafe_member_not_read" for e in entities)
    members = [e for e in entities if e["member_name"] == "log.txt"]
    assert [m["member_index"] for m in members] == [0, 1]
    assert members[0]["sha256"] != members[1]["sha256"]
    cached = next(m for m in matches if m["locator"] == "Log!B1")
    assert cached["raw"]["formula"] == "birth_date"
    assert cached["raw"]["cached_value"] == "0"
    assert "0001" in next(m for m in matches if m["locator"] == "Log!A1")["raw"]["value"]
    assert not any("hidden" in m["raw"].get("text", "") for m in matches)
    assert before == {p.name: sha256_file(p) for p in cfg.source_root.iterdir()}


@pytest.mark.parametrize("failure", ["source_change", "retention", "encoding"])
def test_incomplete_search_is_retained(search_input, failure):
    cfg, inv, policy = search_input
    if failure == "retention":
        policy.write_text(json.dumps({"patterns": {"birth": "birth"}, "max_records": 1}))
    else:
        (cfg.source_root / "fixture.R").write_bytes(
            b"changed" if failure == "source_change" else b"\xff"
        )
        if failure == "encoding":
            updated = new_run(cfg, "updated-inventory")
            inventory(cfg, updated)
            inv = updated / "source_inventory.parquet"
    out = new_run(cfg, "search")
    result = metadata_search(cfg, inv, policy, out)
    assert result["success"] is False
    assert result["failures"] > 0
    assert (out / "failures.json").is_file()
    assert (out / "run.json").is_file()


def test_pdf_page_provenance_without_poppler_in_ci(search_input, monkeypatch):
    cfg, inv, policy = search_input
    (cfg.source_root / "fixture.pdf").write_bytes(b"synthetic PDF placeholder")
    updated = new_run(cfg, "updated-inventory")
    inventory(cfg, updated)
    actual_run = subprocess.run

    def mock_run(args, *positional, **kwargs):
        if args[0] == "pdftotext":
            if args[1:] == ["-v"]:
                return subprocess.CompletedProcess(args, 0, b"", b"synthetic pdftotext version")
            return subprocess.CompletedProcess(args, 0, b"birth page one\fsecond page\f", b"")
        return actual_run(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)
    out = new_run(cfg, "search")
    result = metadata_search(cfg, updated / "source_inventory.parquet", policy, out)
    assert result["success"]
    matches = [json.loads(line) for line in (out / "matches.jsonl").read_text().splitlines()]
    assert any(m["locator"] == "pdf_page:1:line:1" for m in matches)
    pages = json.loads((out / "pdf_text_manifest.json").read_text())
    assert len(pages) == 2
    assert all(sha256_file(out / page["path"]) == page["sha256"] for page in pages)
    entities = json.loads((out / "entities.json").read_text())
    pdf = next(e for e in entities if e["source_path"] == "fixture.pdf")
    assert pdf["pdf_text_tool_version"] == "synthetic pdftotext version"


def test_file_budget_and_output_reuse_are_rejected(search_input):
    cfg, inv, policy = search_input
    policy.write_text(json.dumps({"patterns": {"birth": "birth"}, "max_files": 1}))
    with pytest.raises(ValueError, match="file-count"):
        metadata_search(cfg, inv, policy, new_run(cfg, "search"))
    policy.write_text(json.dumps({"patterns": {"birth": "birth"}}))
    out = new_run(cfg, "search")
    (out / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty new run"):
        metadata_search(cfg, inv, policy, out)
    assert (out / "existing.txt").read_text() == "keep"


def test_text_encodings_and_leading_zeros():
    assert list(text_records("birth,0001".encode("utf-16"), "log.csv")) == [
        ("line:1", "birth,0001")
    ]
    with pytest.raises(ValueError, match="Binary"):
        list(text_records(b"birth\x00", "unknown"))
