"""Read-only source access; immutable per-run output directories."""

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import soundfile as sf

from rat_usv import __version__
from rat_usv.config import HashPolicy, SourceConfig


def safe_source(root: Path, relative: str) -> Path:
    """Reject traversal and all symlink components, including internal links."""
    root = root.resolve(strict=True)
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe source path: {relative!r}")
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink source is not followed: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Not a regular source file: {relative}")
    return resolved


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Hasher:
    def __init__(self, policy: HashPolicy) -> None:
        self.policy = policy
        self.bytes_read = 0

    def compute(self, path: Path) -> tuple[str | None, str]:
        if self.policy.mode == "none":
            return None, "disabled"
        before = path.stat()
        if self.policy.mode == "budget":
            if before.st_size > self.policy.max_file_bytes:
                return None, "skipped_file_limit"
            if self.bytes_read + before.st_size > self.policy.max_total_bytes:
                return None, "skipped_total_budget"
        h = hashlib.sha256()
        with path.open("rb") as handle:
            start = os.fstat(handle.fileno())
            if not stat.S_ISREG(start.st_mode):
                raise ValueError("Only regular files may be hashed")
            remaining = start.st_size
            while remaining:
                block = handle.read(min(1024**2, remaining))
                if not block:
                    break
                self.bytes_read += len(block)
                remaining -= len(block)
                h.update(block)
            end = os.fstat(handle.fileno())
        after = path.stat()

        def signature(s: os.stat_result) -> tuple[int, int, int, int, int]:
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

        if (
            remaining
            or signature(before) != signature(start)
            or signature(start) != signature(end)
            or signature(end) != signature(after)
        ):
            return None, "changed_during_read"
        return h.hexdigest(), "hashed"


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        handle.write("\n")


def new_run(config: SourceConfig, command: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = config.output_root / f"{stamp}-{command}-{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def record_run(
    out: Path, command: str, config: SourceConfig, inputs: list[Path] | None = None
) -> None:
    repo = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    # Code-content digest includes untracked source/config changes before the first commit.
    code = hashlib.sha256()
    snapshot: dict[str, str] = {}
    for folder in (repo / "src", repo / "configs"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != "local.yaml":
                code.update(str(path.relative_to(repo)).encode())
                code.update(path.read_bytes())
                snapshot[str(path.relative_to(repo))] = path.read_text()
    write_json(out / "code_snapshot.json", snapshot)
    write_json(
        out / "run.json",
        {
            "schema_version": 1,
            "run_id": out.name,
            "command": command,
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "libsndfile_version": sf.__libsndfile_version__,
            "dependencies": {
                name: version(name)
                for name in ("numpy", "soundfile", "pyarrow", "pydantic", "typer")
            },
            "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "code_config_sha256": code.hexdigest(),
            "uv_lock_sha256": sha256_file(repo / "uv.lock")
            if (repo / "uv.lock").exists()
            else None,
            "config": config.model_dump(mode="json"),
            "inputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in inputs or []],
            "outputs": [
                {"path": p.name, "sha256": sha256_file(p)}
                for p in sorted(out.iterdir())
                if p.is_file()
            ],
        },
    )
