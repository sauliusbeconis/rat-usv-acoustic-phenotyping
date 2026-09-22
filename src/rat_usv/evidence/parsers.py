"""Read-only, bounded parsers preserving original representations and locators."""

import base64
import io
import posixpath
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from pydantic import JsonValue

Raw = tuple[str, str, dict[str, JsonValue]]
MAX_MEMBER = 8 * 1024**2
MAX_TOTAL = 64 * 1024**2
MAX_ENTRIES = 10000
MAX_CELLS = 200000
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(path.parts) and not (
        path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or "\x00" in name
    )


class BoundedZip:
    """Never extract. Address duplicates by ordinal, never by ambiguous filename."""

    def __init__(self, source: Path | io.BytesIO) -> None:
        self.zip = zipfile.ZipFile(source)
        self.infos = self.zip.infolist()
        self.read_bytes = 0
        if len(self.infos) > MAX_ENTRIES:
            self.zip.close()
            raise ValueError("Archive entry limit exceeded")

    def __enter__(self) -> "BoundedZip":
        return self

    def __exit__(self, *args: object) -> None:
        self.zip.close()

    def read(self, index: int) -> bytes:
        info = self.infos[index]
        if not safe_member(info.filename):
            raise ValueError("Unsafe archive member path")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive symlink is not followed")
        if info.file_size > MAX_MEMBER or self.read_bytes + info.file_size > MAX_TOTAL:
            raise ValueError("Archive read budget exceeded")
        with self.zip.open(info) as stream:
            data = stream.read(min(MAX_MEMBER, MAX_TOTAL - self.read_bytes) + 1)
        self.read_bytes += len(data)
        if len(data) != info.file_size or self.read_bytes > MAX_TOTAL or len(data) > MAX_MEMBER:
            raise ValueError("Archive size mismatch or read budget exceeded")
        return data

    def named(self, name: str) -> bytes:
        matches = [i for i, info in enumerate(self.infos) if info.filename == name]
        if len(matches) != 1:
            raise ValueError(f"Missing/duplicate workbook member: {name}")
        return self.read(matches[0])

    def xml(self, name: str) -> ET.Element:
        data = self.named(name)
        if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ValueError("XML DTD/entities are unsupported")
        return ET.fromstring(data)


def column_number(label: str) -> int:
    result = 0
    for character in label:
        result = result * 26 + ord(character) - 64
    return result


def column_label(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def workbook(data: bytes, name: str) -> Iterator[Raw]:
    if PurePosixPath(name).name.startswith("~$"):
        yield (
            "file",
            "owner_file",
            {"size_bytes": len(data), "raw_base64": base64.b64encode(data).decode()},
        )
        return
    with BoundedZip(io.BytesIO(data)) as package:
        names = {i.filename for i in package.infos}
        strings = []
        if "xl/sharedStrings.xml" in names:
            shared = package.xml("xl/sharedStrings.xml")
            strings = [
                "".join(
                    node.text or ""
                    for child in si
                    for node in (
                        [child]
                        if child.tag == f"{NS}t"
                        else child.findall(f"{NS}t")
                        if child.tag == f"{NS}r"
                        else []
                    )
                )
                for si in shared
            ]
            yield (
                "xl/sharedStrings.xml",
                "workbook_shared_strings",
                {"xml": ET.tostring(shared, encoding="unicode")},
            )
        formats: dict[str, str] = {}
        styles: list[str] = ["0"]
        if "xl/styles.xml" in names:
            style = package.xml("xl/styles.xml")
            formats = {
                n.attrib["numFmtId"]: n.attrib["formatCode"]
                for n in style.findall(f"{NS}numFmts/{NS}numFmt")
            }
            styles = [n.attrib.get("numFmtId", "0") for n in style.findall(f"{NS}cellXfs/{NS}xf")]
            yield (
                "xl/styles.xml",
                "workbook_styles",
                {"xml": ET.tostring(style, encoding="unicode")},
            )
        book = package.xml("xl/workbook.xml")
        yield "xl/workbook.xml", "workbook_metadata", {"xml": ET.tostring(book, encoding="unicode")}
        rels = {r.attrib["Id"]: r.attrib for r in package.xml("xl/_rels/workbook.xml.rels")}
        total_cells = 0
        for sheet in book.findall(f"{NS}sheets/{NS}sheet"):
            relation = rels[
                sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
            ]
            if relation.get("TargetMode") == "External":
                raise ValueError("External worksheet relation is not followed")
            target = relation["Target"]
            member = (
                target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            )
            if not safe_member(member) or not member.startswith("xl/"):
                raise ValueError("Unsafe worksheet relation")
            tree = package.xml(member)
            cell_map: dict[tuple[int, int], ET.Element] = {}
            for element in tree.findall(f"{NS}sheetData/{NS}row/{NS}c"):
                match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", element.attrib.get("r", ""))
                if not match:
                    raise ValueError("Cell has missing/invalid explicit coordinate")
                col, row = column_number(match[1]), int(match[2])
                if (row, col) in cell_map:
                    raise ValueError("Duplicate cell coordinate")
                cell_map[row, col] = element
            rows = max((r for r, c in cell_map), default=0)
            cols = max((c for r, c in cell_map), default=0)
            total_cells += rows * cols
            if total_cells > MAX_CELLS:
                raise ValueError("Worksheet cell rectangle budget exceeded")
            dimension = tree.find(f"{NS}dimension")
            yield (
                f"{sheet.attrib['name']}!sheet",
                "worksheet",
                {
                    "sheet": sheet.attrib["name"],
                    "member": member,
                    "rows": rows,
                    "columns": cols,
                    "dimension": {k: v for k, v in dimension.attrib.items()}
                    if dimension is not None
                    else {},
                    "absence_scope": "A1 through maximum explicitly stored cell; outside is not enumerated",
                },
            )

            def value(cell: ET.Element | None) -> str | None:
                if cell is None:
                    return None
                raw = cell.findtext(f"{NS}v")
                if cell.attrib.get("t") == "s" and raw is not None:
                    index = int(raw)
                    if index < 0 or index >= len(strings):
                        raise ValueError("Invalid shared-string index")
                    return strings[index]
                if cell.attrib.get("t") == "inlineStr":
                    return "".join(n.text or "" for n in cell.findall(f"{NS}is//{NS}t"))
                return raw

            header: list[JsonValue] = [value(cell_map.get((1, col))) for col in range(1, cols + 1)]
            counts: Counter[str] = Counter()
            for row in range(1, rows + 1):
                marker = value(cell_map.get((row, 1)))
                role = "unclassified"
                if row == 1 and marker in {"#", "number", "Number"}:
                    role = "header"
                elif marker in {"min", "max", "mea", "std", "cnt"}:
                    role = "summary"
                elif marker is not None and re.fullmatch(r"[0-9]+", marker):
                    role = "numbered_event_candidate"
                counts[role] += 1
                for col in range(1, cols + 1):
                    cell = cell_map.get((row, col))
                    locator = f"{sheet.attrib['name']}!{column_label(col)}{row}"
                    if cell is None:
                        yield (
                            locator,
                            "cell",
                            {"presence": "absent", "row_role": role, "value": None},
                        )
                        continue
                    v = cell.find(f"{NS}v")
                    formula = cell.find(f"{NS}f")
                    style_index = int(cell.attrib.get("s", "0"))
                    if style_index < 0 or style_index >= len(styles):
                        raise ValueError("Invalid cell style index")
                    fmt = styles[style_index]
                    raw_value = value(cell)
                    yield (
                        locator,
                        "cell",
                        {
                            "presence": "blank"
                            if v is None and raw_value is None and formula is None
                            else "stored",
                            "row_role": role,
                            "cell_type": cell.attrib.get("t", "n"),
                            "stored_v": v.text if v is not None else None,
                            "value": raw_value,
                            "formula_present": formula is not None,
                            "formula": formula.text if formula is not None else None,
                            "formula_attributes": dict(formula.attrib)
                            if formula is not None
                            else {},
                            "cached_value": v.text
                            if v is not None and formula is not None
                            else None,
                            "cache_present": formula is not None and v is not None,
                            "style_index": style_index,
                            "num_fmt_id": fmt,
                            "number_format": formats.get(fmt),
                            "builtin_format_id": fmt if fmt not in formats else None,
                            "xml": ET.tostring(cell, encoding="unicode"),
                        },
                    )
            yield (
                f"{sheet.attrib['name']}!coverage",
                "workbook_coverage",
                {
                    "header": header,
                    "row_counts": dict(counts),
                    "columns": cols,
                    "unit_assignment": None,
                    "formulas_recalculated": False,
                },
            )


def wav(stream: BinaryIO, size: int) -> Iterator[Raw]:
    header = stream.read(12)
    if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
        raise ValueError("Unsupported/truncated RIFF/WAVE header")
    end = struct.unpack_from("<I", header, 4)[0] + 8
    yield (
        "bytes:0:12",
        "riff_header",
        {"raw_hex": header.hex(), "declared_end": end, "file_size": size},
    )
    if end != size:
        raise ValueError("RIFF declared size disagrees with file size")
    offset, metadata_bytes, chunks = 12, 0, 0
    while offset < end:
        chunks += 1
        if chunks > MAX_ENTRIES:
            raise ValueError("RIFF chunk count budget exceeded")
        stream.seek(offset)
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated RIFF chunk header at {offset}")
        tag, length = struct.unpack("<4sI", raw)
        stop = offset + 8 + length
        if stop + length % 2 > end:
            raise ValueError(f"Truncated/oversized RIFF chunk at {offset}")
        locator = f"bytes:{offset}:{8 + length}"
        yield (
            locator,
            "riff_chunk",
            {
                "tag_hex": tag.hex(),
                "tag": tag.decode("ascii", errors="backslashreplace"),
                "size_bytes": length,
                "payload_offset": offset + 8,
            },
        )
        if tag in {b"TIME", b"bext", b"LBL ", b"fmt "}:
            if length > MAX_MEMBER or metadata_bytes + length > MAX_TOTAL:
                raise ValueError(f"RIFF metadata read budget exceeded at {offset}")
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError("Truncated metadata payload")
            metadata_bytes += length
            yield (
                locator,
                "riff_metadata",
                {"tag": tag.decode(), "raw_base64": base64.b64encode(payload).decode()},
            )
            if tag == b"TIME":
                for match in re.finditer(rb"\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", payload):
                    yield (
                        f"bytes:{offset + 8 + match.start()}:{len(match[0])}",
                        "timestamp_text",
                        {
                            "field": "TIME",
                            "value": match[0].decode("ascii"),
                            "timezone": None,
                            "clock_accuracy": None,
                            "date_convention": "unresolved",
                        },
                    )
            elif tag == b"bext":
                if length < 346:
                    raise ValueError("Truncated bext fixed fields")
                for field, start, length_field in (
                    ("origination_date", 320, 10),
                    ("origination_time", 330, 8),
                ):
                    data = payload[start : start + length_field]
                    yield (
                        f"bytes:{offset + 8 + start}:{length_field}",
                        "timestamp_text",
                        {
                            "field": field,
                            "value": data.decode("ascii", errors="backslashreplace"),
                            "raw_hex": data.hex(),
                            "timezone": None,
                            "clock_accuracy": None,
                        },
                    )
            elif tag == b"LBL ":
                position = offset + 8
                for line in payload.splitlines(keepends=True):
                    text = line.decode("latin1")
                    fields = text.replace("\x00", "").strip().split("\t")
                    numeric_pair = len(fields) >= 2 and all(
                        re.fullmatch(r"[+\-]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?", x)
                        for x in fields[:2]
                    )
                    yield (
                        f"bytes:{position}:{len(line)}",
                        "label_line",
                        {
                            "raw_base64": base64.b64encode(line).decode(),
                            "text": text,
                            "candidate_start": fields[0] if numeric_pair else None,
                            "candidate_end": fields[1] if numeric_pair else None,
                            "units": None,
                            "convention": "numeric text only; seconds hypothesis requires review",
                        },
                    )
                    position += len(line)
        offset = stop + length % 2
    yield (
        "file!coverage",
        "wav_coverage",
        {
            "chunks": chunks,
            "metadata_payload_bytes_read": metadata_bytes,
            "audio_payload_bytes_decoded": 0,
            "qc_coverage": None,
        },
    )


def script(data: bytes) -> Iterator[Raw]:
    """Lexical observations only: R is never evaluated and aliases stay strings."""
    offset = 0
    for number, raw in enumerate(data.splitlines(keepends=True), 1):
        yield (
            f"line:{number}",
            "script_line",
            {
                "text": raw.decode("utf-8-sig" if number == 1 else "utf-8"),
                "raw_base64": base64.b64encode(raw).decode(),
                "byte_offset": offset,
            },
        )
        offset += len(raw)


def pdf_pages(path: Path, pages: list[int]) -> Iterator[Raw]:
    version = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    ).stderr.strip()
    for page in pages:
        result = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(path), "-"],
            capture_output=True,
            check=True,
            timeout=30,
        )
        if len(result.stdout) > MAX_MEMBER:
            raise ValueError("PDF page text budget exceeded")
        yield (
            f"pdf_page:{page}",
            "pdf_page_text",
            {
                "text": result.stdout.decode("utf-8"),
                "extractor": version,
                "page_numbering": "one-based PDF page",
                "figures_transcribed": False,
            },
        )


def archive_directory(package: BoundedZip) -> Iterator[Raw]:
    counts = Counter(i.filename for i in package.infos)
    for ordinal, info in enumerate(package.infos):
        yield (
            f"zip_entry:{ordinal}",
            "archive_entry",
            {
                "member_name": info.filename,
                "member_index": ordinal,
                "header_offset": info.header_offset,
                "size_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": info.CRC,
                "is_directory": info.is_dir(),
                "safe_path": safe_member(info.filename),
                "duplicate_name_count": counts[info.filename],
                "content_read_by_directory_scan": False,
            },
        )
