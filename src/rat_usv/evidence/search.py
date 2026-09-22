"""Bounded metadata discovery; matches are source observations, never confirmed identities."""

import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from rat_usv.config import SourceConfig
from rat_usv.evidence import parsers
from rat_usv.evidence.pilot import signature
from rat_usv.provenance.io import record_run, safe_source, sha256_file, write_json
from rat_usv.schemas.evidence import stable_id

SEARCH_VERSION = "metadata-search/1"
TEXT_SUFFIXES = {
    ".r",
    ".rmd",
    ".txt",
    ".csv",
    ".json",
    ".html",
    ".ini",
    ".pper",
    ".error",
    ".rproj",
    "",
}
SERIALIZED_SUFFIXES = {".rdata", ".rds", ".rdb", ".rdx"}


class SearchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    patterns: dict[str, str] = Field(min_length=1)
    max_files: int = Field(default=3000, gt=0, le=10000)
    max_file_bytes: int = Field(default=16 * 1024**2, gt=0, le=64 * 1024**2)
    max_source_bytes: int = Field(default=128 * 1024**2, gt=0, le=512 * 1024**2)
    max_records: int = Field(default=100000, gt=0, le=1000000)
    max_pdf_pages: int = Field(default=300, gt=0, le=1000)


class VisibleHtml(HTMLParser):
    """Read text outside script/style tags without fetching or rendering active HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.rows: list[tuple[int, int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            line, column = self.getpos()
            self.rows.append((line, column, data))


def text_records(data: bytes, name: str) -> Iterator[tuple[str, str]]:
    """Retain exact text and line positions; no CSV numeric inference or R execution."""
    text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    if "\x00" in text:
        raise ValueError("Binary or unsupported text encoding")
    if Path(name).suffix.lower() == ".html":
        html = VisibleHtml()
        html.feed(text)
        for line, column, value in html.rows:
            yield f"html_line:{line}:column:{column}", value
    else:
        for line, value in enumerate(text.splitlines(), 1):
            yield f"line:{line}", value


def metadata_search(
    config: SourceConfig, inventory_path: Path, policy_path: Path, out: Path
) -> dict[str, Any]:
    """Search a hash-bearing inventory and supported ZIP members, retaining coverage gaps."""
    policy = SearchPolicy.model_validate_json(policy_path.read_bytes())
    patterns = {
        label: re.compile(pattern, re.IGNORECASE) for label, pattern in policy.patterns.items()
    }
    rows = pq.read_table(inventory_path).to_pylist()
    if len(rows) > policy.max_files:
        raise ValueError("Metadata search file-count budget exceeded")
    if len({r["relative_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate inventory paths")
    out = out.resolve(strict=True)
    if (
        out == config.output_root
        or not out.is_relative_to(config.output_root)
        or any(out.iterdir())
    ):
        raise ValueError("Metadata search requires an empty new run below output_root")
    selected = []
    for row in rows:
        path = safe_source(config.source_root, row["relative_path"])
        if any(
            path.is_relative_to((config.source_root / x).resolve()) for x in config.exclude_relative
        ):
            raise ValueError("Inventory selects an excluded subtree")
        if row.get("hash_status") != "hashed" or not row.get("sha256"):
            raise ValueError("Metadata search requires hashes for every inventory source")
        if path.stat().st_size > policy.max_file_bytes:
            raise ValueError("Metadata search per-file byte budget exceeded")
        selected.append((row, path))
    if sum(path.stat().st_size for _, path in selected) > policy.max_source_bytes:
        raise ValueError("Metadata search total byte budget exceeded")
    write_json(out / "policy.json", policy.model_dump())
    write_json(out / "input_inventory.json", rows)
    entities: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    verification: list[dict[str, Any]] = []
    retained = 0
    bytes_read = 0
    match_counts: Counter[str] = Counter()
    (out / "pdf_text").mkdir()

    with (out / "matches.jsonl").open("x", encoding="utf-8") as stream:

        def emit(entity: dict[str, Any], locator: str, kind: str, raw: dict[str, Any]) -> None:
            nonlocal retained
            serialized = json.dumps(raw, ensure_ascii=False)
            labels = [label for label, pattern in patterns.items() if pattern.search(serialized)]
            if not labels:
                return
            if retained >= policy.max_records:
                raise ValueError("Metadata match retention budget exceeded")
            value = {
                "match_id": stable_id("metadata-match", [entity["entity_id"], locator, kind, raw]),
                "entity_id": entity["entity_id"],
                "locator": locator,
                "kind": kind,
                "raw": raw,
                "matched_categories": labels,
                "interpretation": "search hit only; applicability and biological meaning unreviewed",
            }
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            retained += 1
            match_counts.update(labels)
            entity["matches"] += 1

        def inspect(data: bytes, entity: dict[str, Any], path: Path | None = None) -> None:
            name = entity["member_name"] or entity["source_path"]
            suffix = Path(name).suffix.lower()
            entity["matches"] = 0
            entity["records_inspected"] = 0
            entities.append(entity)
            if suffix in SERIALIZED_SUFFIXES or Path(name).name.lower() == ".rdata":
                entity["status"] = "not_deserialized"
                return
            if suffix == ".xlsx":
                entity["worksheet_coverage"] = []
                for locator, kind, raw in parsers.workbook(data, name):
                    entity["records_inspected"] += 1
                    if kind == "workbook_coverage":
                        entity["worksheet_coverage"].append({"locator": locator, **raw})
                    if kind in {"cell", "workbook_metadata", "worksheet"}:
                        emit(entity, locator, kind, raw)
                    if kind == "owner_file":
                        entity["status"] = "owner_file_not_workbook"
                entity.setdefault("status", "searched_cells_formulas_and_caches")
            elif suffix == ".pdf" and path is not None:
                version = subprocess.run(
                    ["pdftotext", "-v"], capture_output=True, check=True, timeout=10
                )
                entity["pdf_text_tool_version"] = (version.stdout + version.stderr).decode(
                    "utf-8", errors="replace"
                )
                result = subprocess.run(
                    ["pdftotext", "-layout", str(path), "-"],
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
                if len(result.stdout) > parsers.MAX_MEMBER:
                    raise ValueError("PDF text byte budget exceeded")
                pages = result.stdout.decode("utf-8").split("\f")
                if pages and not pages[-1].strip():
                    pages.pop()
                if len(pages) > policy.max_pdf_pages:
                    raise ValueError("PDF page budget exceeded")
                text_dir = out / "pdf_text" / entity["entity_id"]
                text_dir.mkdir()
                for page, value in enumerate(pages, 1):
                    with (text_dir / f"page-{page:03d}.txt").open("x", encoding="utf-8") as handle:
                        handle.write(value)
                    for line, text in enumerate(value.splitlines(), 1):
                        entity["records_inspected"] += 1
                        emit(
                            entity, f"pdf_page:{page}:line:{line}", "pdf_text_line", {"text": text}
                        )
                entity["pages"] = len(pages)
                entity["status"] = "searched_pdf_text_figures_untranscribed"
            elif suffix in TEXT_SUFFIXES:
                entity["text_encoding"] = (
                    "utf-16-bom" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
                )
                for locator, text in text_records(data, name):
                    entity["records_inspected"] += 1
                    emit(entity, locator, "text", {"text": text})
                entity["status"] = "searched_text_without_execution"
            elif suffix == ".zip" and path is not None:
                with parsers.BoundedZip(io.BytesIO(data)) as archive:
                    for ordinal, member in enumerate(archive.infos):
                        child = {
                            "entity_id": stable_id(
                                "metadata-source", [entity["entity_id"], ordinal, member.filename]
                            ),
                            "source_path": entity["source_path"],
                            "container_sha256": entity["sha256"],
                            "member_name": member.filename,
                            "member_index": ordinal,
                            "size_bytes": member.file_size,
                            "sha256": None,
                            "matches": 0,
                            "records_inspected": 0,
                        }
                        if member.is_dir() or not parsers.safe_member(member.filename):
                            child["status"] = (
                                "directory" if member.is_dir() else "unsafe_member_not_read"
                            )
                            entities.append(child)
                            continue
                        try:
                            content = archive.read(ordinal)
                            child["sha256"] = hashlib.sha256(content).hexdigest()
                            inspect(content, child)
                        except Exception as error:
                            child["status"] = "inspection_error"
                            child["error"] = f"{type(error).__name__}: {error}"
                            if child not in entities:
                                entities.append(child)
                            failures.append(
                                {"entity_id": child["entity_id"], "error": child["error"]}
                            )
                entity["status"] = "searched_archive_members_no_extraction"
            else:
                entity["status"] = "unsupported_format_not_inspected"

        for row, path in selected:
            before = signature(path)
            entity = {
                "entity_id": stable_id("metadata-source", [row["relative_path"], row["sha256"]]),
                "source_path": row["relative_path"],
                "sha256": row["sha256"],
                "size_bytes": before["size_bytes"],
                "member_name": None,
                "member_index": None,
                "container_sha256": None,
            }
            try:
                with path.open("rb") as handle:
                    data = handle.read(policy.max_file_bytes + 1)
                bytes_read += len(data)
                if len(data) > policy.max_file_bytes or bytes_read > policy.max_source_bytes:
                    raise ValueError("Metadata search byte budget exceeded during read")
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("Source differs from inventory hash")
                inspect(data, entity, path)
            except Exception as error:
                entity["status"] = "inspection_error"
                entity["error"] = f"{type(error).__name__}: {error}"
                if entity not in entities:
                    entities.append(entity)
                failures.append({"entity_id": entity["entity_id"], "error": entity["error"]})
            finally:
                try:
                    digest = sha256_file(path)
                    after = signature(path)
                    unchanged = before == after and digest == row["sha256"]
                    verification.append(
                        {
                            "source_path": row["relative_path"],
                            "before": before,
                            "after": after,
                            "after_sha256": digest,
                            "unchanged": unchanged,
                        }
                    )
                    if not unchanged:
                        failures.append(
                            {
                                "entity_id": entity["entity_id"],
                                "error": "Source changed during search",
                            }
                        )
                except OSError as error:
                    failures.append({"entity_id": entity["entity_id"], "error": str(error)})
    write_json(out / "entities.json", entities)
    write_json(out / "verification.json", verification)
    write_json(out / "failures.json", failures)
    write_json(
        out / "pdf_text_manifest.json",
        [
            {"path": str(p.relative_to(out)), "sha256": sha256_file(p)}
            for p in sorted((out / "pdf_text").rglob("*.txt"))
        ],
    )
    summary = {
        "search_version": SEARCH_VERSION,
        "source_files": len(rows),
        "entities_including_members": len(entities),
        "source_bytes_read": bytes_read,
        "matches": retained,
        "match_categories": dict(match_counts),
        "status_counts": dict(Counter(e["status"] for e in entities)),
        "failures": len(failures),
        "success": not failures,
        "source_verification_passed": len(verification) == len(rows)
        and all(v["unchanged"] for v in verification),
        "canonical_assignments": 0,
        "limits": [
            "Search hits are not scientific confirmations",
            "Keyword absence is not evidence of record absence",
            "Images, embedded objects and serialized R objects are not interpreted",
            "Text outside supported encodings is an explicit error",
            "Copied sources are retained without assuming independence",
        ],
    }
    write_json(out / "summary.json", summary)
    record_run(out, "metadata-search", config, [inventory_path, policy_path])
    return summary
