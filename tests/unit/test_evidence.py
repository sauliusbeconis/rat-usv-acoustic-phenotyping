"""Synthetic-only evidence fixtures; historical files are never CI inputs."""

import io
import json
import struct
import warnings
import zipfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from rat_usv.cli import app
from rat_usv.config import SourceConfig
from rat_usv.evidence import parsers
from rat_usv.evidence.pilot import (
    extract_pilot,
    script_assertions,
    timestamp_review,
    validate_bundle_file,
)
from rat_usv.provenance.io import new_run, sha256_file
from rat_usv.schemas.evidence import Artifact, Assertion, EvidenceBundle, Observation, stable_id


def synthetic_workbook(columns=17, hostile_dimension=False):
    def cell(ref, value=None, kind=None, extra=""):
        attr = f' t="{kind}"' if kind else ""
        body = (
            ""
            if value is None
            else (f"<is><t>{value}</t></is>" if kind == "inlineStr" else f"<v>{value}</v>")
        )
        return f'<c r="{ref}"{attr}>{extra}{body}</c>'

    headers = ["#", "label", "class", "duration", "break", "start time", "end time"] + [
        f"frequency{i}" for i in range(10)
    ]
    if columns == 16:
        headers.pop(1)
    header = "".join(
        cell(f"{parsers.column_label(i)}1", v, "inlineStr") for i, v in enumerate(headers, 1)
    )
    rows = [f'<row r="1">{header}</row>']
    for i, label in enumerate(("min", "max", "mea", "std", "cnt"), 2):
        rows.append(
            f'<row r="{i}">{cell(f"A{i}", label, "inlineStr")}{cell(f"B{i}", "139")}{cell(f"D{i}", "139000")}{cell(f"H{i}", "0.139")}</row>'
        )
    rows.append(
        f'<row r="7">{cell("A7", "1")}{cell("B7", "_", "inlineStr")}{cell("C7")}{cell("D7", "0")}{cell("F7", "26.200000000000003")}{cell("G7", "105.7")}</row>'
    )
    rows.append(
        f'<row r="8">{cell("A8", "2")}{cell("B8", "7", extra="<f>1+1</f>")}{cell("C8", "0001", "inlineStr")}{cell("D8", "0011u", "inlineStr")}{cell("E8", "", "inlineStr")}</row>'
    )
    if hostile_dimension:
        rows.append('<row r="900000"><c r="XFD900000"><v>1</v></c></row>')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{parsers.NS[1:-1]}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="r1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        z.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{parsers.NS[1:-1]}"><sheetData>{"".join(rows)}</sheetData></worksheet>',
        )
    return stream.getvalue()


def synthetic_wav(chunks):
    data = b"WAVE" + b"".join(
        tag + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")
        for tag, payload in chunks
    )
    return b"RIFF" + struct.pack("<I", len(data)) + data


@pytest.mark.parametrize("columns", [16, 17])
def test_workbook_original_representations(columns):
    rows = list(parsers.workbook(synthetic_workbook(columns), "source.xlsx"))
    cells = {loc: raw for loc, kind, raw in rows if kind == "cell"}
    assert cells["Sheet1!B7"]["value"] == "_"
    assert cells["Sheet1!C7"]["presence"] == "blank"
    assert cells["Sheet1!E7"]["presence"] == "absent"
    assert cells["Sheet1!D7"]["stored_v"] == "0"
    assert cells["Sheet1!E8"]["value"] == ""
    assert cells["Sheet1!F7"]["stored_v"] == "26.200000000000003"
    assert cells["Sheet1!B8"]["formula"] == "1+1"
    assert cells["Sheet1!B8"]["cached_value"] == "7"  # never recalculated
    assert cells["Sheet1!C8"]["value"] == "0001"
    assert cells["Sheet1!D8"]["value"] == "0011u"
    assert cells["Sheet1!D6"]["value"] == "139000"  # never scaled
    assert cells["Sheet1!H6"]["value"] == "0.139"
    coverage = next(raw for _, kind, raw in rows if kind == "workbook_coverage")
    assert coverage["columns"] == columns
    assert coverage["row_counts"] == {"header": 1, "summary": 5, "numbered_event_candidate": 2}


def test_owner_file_and_oversized_worksheet():
    assert list(parsers.workbook(b"not a workbook", "~$fixture.xlsx"))[0][1] == "owner_file"
    with pytest.raises(ValueError, match="rectangle budget"):
        list(parsers.workbook(synthetic_workbook(hostile_dimension=True), "fixture.xlsx"))
    with pytest.raises(zipfile.BadZipFile):
        list(parsers.workbook(b"not a workbook", "fixture.xlsx"))


def test_riff_directory_skips_audio_and_preserves_dates():
    bext = bytearray(346)
    bext[320:330], bext[330:338] = b"2020-01-14", b"12:30:00"
    data = synthetic_wav(
        [
            (b"JUNK", b"x"),
            (b"data", b"\x00" * 1000),
            (b"TIME", b"01/13/20 12:30:00"),
            (b"bext", bext),
            (b"LBL ", b"header\r\n0.02\x00\t0.10\x00\t1\r\n"),
        ]
    )

    class Meter(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.bytes_read = 0

        def read(self, n=-1):
            assert n >= 0
            value = super().read(n)
            self.bytes_read += len(value)
            return value

    stream = Meter(data)
    rows = list(parsers.wav(stream, len(data)))
    assert stream.bytes_read < len(data) - 1000
    timestamps = [r["value"] for _, k, r in rows if k == "timestamp_text"]
    assert "01/13/20 12:30:00" in timestamps and "2020-01-14" in timestamps
    labels = [r for _, k, r in rows if k == "label_line"]
    assert labels[-1]["candidate_start"] == "0.02"
    assert rows[-1][2]["audio_payload_bytes_decoded"] == 0
    _, observations = artifact_and_observations("synthetic.wav", rows)
    comparison = timestamp_review(observations)
    assert comparison["status"] == "CONFLICT"
    assert comparison["timezone"] is None


@pytest.mark.parametrize(
    "data",
    [
        b"RIFF",
        b"RIFF\xff\xff\xff\xffWAVE",
        b"RIFF\x06\x00\x00\x00WAVEab",
        b"RIFF\x0c\x00\x00\x00WAVEJUNK\xff\xff\xff\xff",
    ],
)
def test_riff_explicit_failures(data):
    with pytest.raises(ValueError):
        list(parsers.wav(io.BytesIO(data), len(data)))


def test_absent_labels_and_metadata_budget(monkeypatch):
    data = synthetic_wav([(b"data", b"00")])
    assert not any(k == "label_line" for _, k, _ in parsers.wav(io.BytesIO(data), len(data)))
    monkeypatch.setattr(parsers, "MAX_MEMBER", 3)
    data = synthetic_wav([(b"TIME", b"long")])
    with pytest.raises(ValueError, match="budget"):
        list(parsers.wav(io.BytesIO(data), len(data)))


def test_archive_duplicate_names_traversal_and_budgets(monkeypatch):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("same.txt", b"first")
            archive.writestr("same.txt", b"second")
            archive.writestr("../escape.txt", b"forbidden")
            archive.writestr("/absolute.txt", b"forbidden")
    with parsers.BoundedZip(io.BytesIO(stream.getvalue())) as archive:
        assert archive.read(0) == b"first" and archive.read(1) == b"second"
        rows = list(parsers.archive_directory(archive))
        assert rows[0][2]["duplicate_name_count"] == 2
        for index in (2, 3):
            with pytest.raises(ValueError, match="Unsafe"):
                archive.read(index)
        with pytest.raises(ValueError, match="duplicate"):
            archive.named("same.txt")
        monkeypatch.setattr(parsers, "MAX_TOTAL", 12)
        with pytest.raises(ValueError, match="budget"):
            archive.read(0)


def artifact_and_observations(source, raw_rows, digest="a" * 64, lineage="historical"):
    art = Artifact(
        artifact_id=stable_id("artifact", [source, None, None, digest]),
        source_path=source,
        sha256=digest,
        size_bytes=1,
        lineage=lineage,
    )
    obs = [
        Observation(
            observation_id=stable_id("obs", [art.artifact_id, loc, kind, raw, "m0.1/2"]),
            artifact_id=art.artifact_id,
            locator=loc,
            kind=kind,
            raw=raw,
            manifest_sha256="b" * 64,
        )
        for loc, kind, raw in raw_rows
    ]
    return art, obs


def test_lexical_recodes_preserve_scope_zeros_and_quotedness():
    data = b"""Ignored prose\n```{r}\npateles_1_vada <- filter(x, vados_nr == 1 & individo_id %in% c("0001", "0011u"))\npatinai_2_vada <- filter(x, vados_nr == 2 & individo_id %in% c("0001"))\npateles_3_vada <- filter(x, vados_nr == "3" & individo_id %in% c(0001))\na <- fct_recode(\n "16" = "01.12",\n "17" = "01.14"\n)\n```\n"""
    art, obs = artifact_and_observations("source.Rmd", parsers.script(data))
    values = script_assertions(art, obs)
    assert len(values) == 6
    assert values[0].context["alias"] == "0001"
    assert values[1].context["alias"] == "0011u"
    assert values[0].subject != values[2].subject
    assert values[3].context["leading_zero_risk"] is True
    assert [v.context["date_token"] for v in values if v.field == "historical_day_label"] == [
        "01.12",
        "01.14",
    ]
    assert all(a.canonical_id is None and a.status == "unresolved" for a in values)


def test_pilot_determinism_hashes_members_and_failures(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.xlsx").write_bytes(synthetic_workbook())
    (source / "b.xlsx").write_bytes((source / "a.xlsx").read_bytes())
    (source / "~$a.xlsx").write_bytes(b"owner")
    (source / "0001.wav").write_bytes(
        synthetic_wav([(b"TIME", b"01/14/20 01:02:03"), (b"data", b"00")])
    )
    with zipfile.ZipFile(source / "a.zip", "w") as archive:
        archive.writestr("copy.xlsx", (source / "a.xlsx").read_bytes())
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic protocol v0")
    config = SourceConfig(source_root=source, output_root=tmp_path / "runs")
    selection = tmp_path / "pilot.yaml"
    selected = {
        "protocol": "protocol.md",
        "sources": [
            {
                "path": p.name,
                "kind": "archive"
                if p.suffix == ".zip"
                else "wav"
                if p.suffix == ".wav"
                else "workbook",
                "selection_reason": "synthetic",
                **({"members": ["copy.xlsx"]} if p.suffix == ".zip" else {}),
            }
            for p in sorted(source.iterdir())
        ],
        "candidate_links": [
            {
                "recording": "0001.wav",
                "export": "a.xlsx",
                "reason": "synthetic unresolved association",
            }
        ],
    }
    selection.write_text(yaml.safe_dump(selected))
    before = {p: sha256_file(p) for p in source.iterdir()}
    first, second = new_run(config, "pilot"), new_run(config, "pilot")
    for out in (first, second):
        report = extract_pilot(config, selection, out)
        assert report["success"] and report["source_verification_passed"]
        assert report["canonical_biological_assignments"] == 0
    assert (first / "evidence.json").read_bytes() == (second / "evidence.json").read_bytes()
    assert before == {p: sha256_file(p) for p in before}
    data = json.loads((first / "evidence.json").read_text())
    bundle = EvidenceBundle.model_validate(data)
    assert validate_bundle_file(first / "evidence.json") == bundle
    cli = CliRunner().invoke(app, ["validate-evidence", "--bundle", str(first / "evidence.json")])
    assert cli.exit_code == 0 and '"m0_ready": false' in cli.stdout
    member = next(a for a in bundle.artifacts if a.member_name)
    assert member.member_index == 0 and member.container_sha256 == before[source / "a.zip"]
    assert len(report["duplicate_groups"][0]) == 3
    assert bundle.relationships[0].canonical_id is None
    with pytest.raises(ValueError, match="empty new run"):
        extract_pilot(config, selection, first)
    (source / "broken.xlsx").write_bytes(b"invalid zip")
    selected["sources"].append(
        {"path": "broken.xlsx", "kind": "workbook", "selection_reason": "synthetic failure"}
    )
    selection.write_text(yaml.safe_dump(selected))
    failed = new_run(config, "pilot")
    assert extract_pilot(config, selection, failed)["failure_count"] == 1
    assert json.loads((failed / "failures.json").read_text())[0]["stage"] == "parser"
    assert (failed / "run.json").is_file()
    (first / "source_manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="manifest digest"):
        validate_bundle_file(first / "evidence.json")


def test_source_output_separation_and_failed_cli(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValidationError, match="excluded subtree"):
        SourceConfig(source_root=source, output_root=source / "output")
    (source / "out").mkdir()
    (source / "out" / "data.xlsx").write_bytes(b"derivative")
    cfg = SourceConfig(source_root=source, output_root=source / "out", exclude_relative=["out"])
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic")
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "protocol": "protocol.md",
                "sources": [
                    {"path": "out/data.xlsx", "kind": "workbook", "selection_reason": "invalid"}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="excluded/output"):
        extract_pilot(cfg, selection, new_run(cfg, "pilot"))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    result = CliRunner().invoke(
        app, ["evidence-pilot", "--config", str(config), "--selection", str(selection)]
    )
    assert result.exit_code == 1
    out = Path(result.stdout.strip())
    assert (out / "fatal_failure.json").is_file() and (out / "run.json").is_file()


def test_strict_hash_json_and_observation_integrity():
    art, obs = artifact_and_observations("a", [("line:1", "script_line", {"text": "original"})])
    bundle = EvidenceBundle(
        manifest_sha256="b" * 64, protocol_sha256="c" * 64, artifacts=[art], observations=obs
    )
    raw = bundle.model_dump()
    raw["observations"][0]["raw"]["text"] = "tampered"
    with pytest.raises(ValidationError, match="content ID"):
        EvidenceBundle.model_validate(raw)
    with pytest.raises(ValidationError):
        Artifact.model_validate({**art.model_dump(), "sha256": "not a hash"})
    with pytest.raises(ValidationError):
        Assertion(
            assertion_id="a",
            subject="s",
            field="weight",
            value=3,
            observation_ids=[obs[0].observation_id],
        )
    with pytest.raises(ValidationError):
        Assertion(
            assertion_id="a",
            subject="s",
            field="sex",
            value="female",
            observation_ids=[obs[0].observation_id],
            canonical_id="invented",
        )


def test_source_change_during_extraction_is_failure(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    original = root / "a.xlsx"
    original.write_bytes(synthetic_workbook())
    config = SourceConfig(source_root=root, output_root=tmp_path / "runs")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic")
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "protocol": "protocol.md",
                "sources": [
                    {
                        "path": "a.xlsx",
                        "kind": "workbook",
                        "selection_reason": "synthetic mutation test",
                    }
                ],
            }
        )
    )
    parser = parsers.workbook

    def mutate_fixture(data, name):
        yield from parser(data, name)
        original.write_bytes(b"synthetic concurrent modification")

    monkeypatch.setattr(parsers, "workbook", mutate_fixture)
    out = new_run(config, "pilot")
    report = extract_pilot(config, selection, out)
    assert not report["success"] and not report["source_verification_passed"]
    assert any(
        r["stage"] == "verification" for r in json.loads((out / "failures.json").read_text())
    )


def test_xml_entities_duplicate_cells_and_external_links_rejected():
    original = synthetic_workbook()
    with zipfile.ZipFile(io.BytesIO(original)) as package:
        entries = {i.filename: package.read(i) for i in package.infolist()}
    attacks = [
        ("xl/workbook.xml", b'<!DOCTYPE workbook [<!ENTITY x "hidden">]><workbook/>', "DTD"),
        (
            "xl/_rels/workbook.xml.rels",
            b'<Relationships><Relationship Id="r1" Target="https://example.invalid" TargetMode="External"/></Relationships>',
            "External",
        ),
        (
            "xl/worksheets/sheet1.xml",
            entries["xl/worksheets/sheet1.xml"].replace(
                b'<row r="1">', b'<row r="1"><c r="A1"><v>2</v></c>'
            ),
            "Duplicate",
        ),
    ]
    for name, payload, match in attacks:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as package:
            for member, data in entries.items():
                package.writestr(member, payload if member == name else data)
        with pytest.raises(ValueError, match=match):
            list(parsers.workbook(stream.getvalue(), "synthetic.xlsx"))
