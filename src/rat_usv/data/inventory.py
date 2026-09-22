"""Filesystem inventory without following symlinks or opening source for writing."""

import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rat_usv.config import SourceConfig
from rat_usv.provenance.io import Hasher, record_run, write_json

INVENTORY_SCHEMA = pa.schema(
    [
        ("relative_path", pa.string()),
        ("file_type", pa.string()),
        ("size_bytes", pa.int64()),
        ("mtime_ns", pa.int64()),
        ("sha256", pa.string()),
        ("hash_status", pa.string()),
        ("role", pa.string()),
        ("parser_status", pa.string()),
        ("notes", pa.string()),
    ]
)


def role(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".wav":
        return "candidate_raw_audio"
    if ext in {".xlsx", ".xls", ".csv", ".tsv"}:
        return "candidate_metadata_or_acoustic_export"
    if ext in {".r", ".rmd", ".rproj", ".rdata", ".rds"}:
        return "historical_analysis"
    if ext in {".pdf", ".docx", ".doc"}:
        return "reference_document"
    if ext in {".zip", ".7z", ".tar"}:
        return "archive_not_expanded"
    if ext in {".ini", ".pper"}:
        return "candidate_acquisition_or_application_metadata"
    return "unclassified"


def inventory(config: SourceConfig, out: Path) -> list[dict[str, Any]]:
    root = config.source_root
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped: list[dict[str, str]] = []
    exclude = [Path(x) for x in config.exclude_relative]

    def excluded(path: Path) -> bool:
        rel = path.relative_to(root)
        return any(rel == x or rel.is_relative_to(x) for x in exclude)

    for base, dirs, names in os.walk(
        root, followlinks=False, onerror=lambda e: errors.append(str(e))
    ):
        current = Path(base)
        keep = []
        for name in sorted(dirs):
            path = current / name
            if path.is_symlink() or excluded(path):
                skipped.append(
                    {
                        "path": str(path.relative_to(root)),
                        "reason": "symlink" if path.is_symlink() else "configured_exclusion",
                    }
                )
            else:
                keep.append(name)
        dirs[:] = keep
        for name in sorted(names):
            path = current / name
            if excluded(path):
                skipped.append(
                    {"path": str(path.relative_to(root)), "reason": "configured_exclusion"}
                )
                continue
            row: dict[str, Any] = {
                "relative_path": str(path.relative_to(root)),
                "file_type": path.suffix.lower() or "[no_extension]",
                "size_bytes": None,
                "mtime_ns": None,
                "sha256": None,
                "hash_status": "not_attempted",
                "role": role(path),
                "parser_status": "not_parsed",
                "notes": "Role is a filename-based candidate, not biological evidence.",
            }
            try:
                st = path.lstat()
                row.update(size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
                if not stat.S_ISREG(st.st_mode):
                    row.update(hash_status="skipped_nonregular", parser_status="skipped_nonregular")
            except OSError as e:
                row.update(parser_status="stat_error", notes=str(e))
                errors.append(str(e))
            rows.append(row)
    # Small historical evidence is prioritized; deterministic path order within each class.
    rows.sort(key=lambda r: (r["file_type"] == ".wav", r["relative_path"]))
    hasher = Hasher(config.hash)
    for row in rows:
        if row["hash_status"] != "not_attempted" or row["parser_status"] == "stat_error":
            continue
        try:
            row["sha256"], row["hash_status"] = hasher.compute(root / row["relative_path"])
        except (OSError, ValueError) as e:
            row.update(hash_status="hash_error", notes=str(e))
            errors.append(str(e))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=INVENTORY_SCHEMA), out / "source_inventory.parquet"
    )
    # JSON quoting preserves spaces, newlines and aliases. This is not sha256sum syntax.
    write_json(
        out / "checksums.json", {r["relative_path"]: r["sha256"] for r in rows if r["sha256"]}
    )
    write_json(
        out / "summary.json",
        {
            "files": len(rows),
            "bytes": sum(r["size_bytes"] or 0 for r in rows),
            "types": dict(Counter(r["file_type"] for r in rows)),
            "hash_status": dict(Counter(r["hash_status"] for r in rows)),
            "hash_bytes_read": hasher.bytes_read,
            "traversal_errors": errors,
            "skipped_subtrees_or_paths": skipped,
            "complete_traversal": not errors,
        },
    )
    record_run(out, "inventory", config)
    return rows
