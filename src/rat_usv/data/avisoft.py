"""Read 2020 Avisoft SASLab call exports (xlsx in the archive) as a detection reference."""

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from rat_usv.data.linkage import Recording

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
# E2L2-1000 workbooks for these dates are rotated against the WAVs (save-time pairing).
ROTATED = {("E2L2-1000", "2019-12-29"), ("E2L2-1000", "2019-12-30"), ("E2L2-1000", "2019-12-31")}


@dataclass(frozen=True)
class ReferenceCall:
    start_s: float
    end_s: float
    label: str | None


def _cells(data: bytes) -> list[dict[str, str | None]]:
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.iter(f"{NS}t")) for si in root]
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{NS}row"):
        values: dict[str, str | None] = {}
        for cell in row.findall(f"{NS}c"):
            node = cell.find(f"{NS}v")
            value = None if node is None else node.text
            if cell.get("t") == "s" and value is not None:
                value = strings[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in cell.iter(f"{NS}t"))
            values[re.sub(r"\d", "", cell.get("r", ""))] = value
        rows.append(values)
    return rows


def parse_export(data: bytes) -> list[ReferenceCall]:
    """Numbered call rows only (summary rows such as min/max are skipped); times in ms."""
    rows = _cells(data)
    if not rows:
        return []
    column = {str(v).strip(): k for k, v in rows[0].items() if v}
    if "start time" not in column or "end time" not in column:
        raise ValueError("Not an Avisoft call export: missing start/end time columns")
    label_col = column.get("class") or column.get("label")
    calls = []
    for row in rows[1:]:
        number = row.get(column.get("#", "A"))
        if number is None or not re.fullmatch(r"\d+(\.0+)?", number):
            continue
        start, end = row.get(column["start time"]), row.get(column["end time"])
        if start is None or end is None:
            continue
        calls.append(
            ReferenceCall(
                float(start) / 1000, float(end) / 1000, row.get(label_col) if label_col else None
            )
        )
    return calls


def archive_member(rec: Recording) -> str | None:
    """Archive path of the export for a final pup recording, or None if not mapped."""
    if not rec.is_final_pup or rec.recording_date is None or rec.animal_id is None:
        return None
    if (rec.animal_id, rec.recording_date.isoformat()) in ROTATED:
        return None
    code = rec.animal_id.split("-")[1]
    litter = rec.litter_id
    if litter == "E2L2":
        return f"Data/duomenys/2_4/{code}/{rec.postnatal_age_days}.xlsx"
    group = {"E2L1": "1_10", "E2L3": "3_14"}.get(litter or "")
    if group is None:
        return None
    return f"Data/duomenys/{group}/{code}/{rec.recording_date:%m.%d}.xlsx"


def load_references(archive: Path, recordings: list[Recording]) -> dict[str, list[ReferenceCall]]:
    """Map wav_path -> reference calls for every recording whose export exists."""
    out: dict[str, list[ReferenceCall]] = {}
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        for rec in recordings:
            member = archive_member(rec)
            if member and member in names:
                out[rec.wav_path] = parse_export(package.read(member))
    return out
