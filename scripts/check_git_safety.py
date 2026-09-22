"""Reject raw/private artifacts even if force-added; inspect index blobs on commit."""

import argparse
import subprocess
from pathlib import PurePosixPath

BANNED = {
    ".wav",
    ".flac",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".rdata",
    ".rds",
    ".rhistory",
    ".parquet",
    ".arrow",
    ".feather",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".zip",
    ".7z",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".pdf",
    ".docx",
}
BLOCK_DIRS = {
    ".local",
    ".agents",
    ".codex",
    "raw",
    "private",
    "derived",
    "mlruns",
    "mlartifacts",
    ".venv",
    ".git",
    ".rproj.user",
}
ALLOWED_CSV = {"figures/figure_registry.csv"}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true")
    group.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if args.staged
        else ["git", "ls-files", "-z"]
    )
    files = subprocess.check_output(command).decode().split("\0")
    errors = []
    for name in filter(None, files):
        path = PurePosixPath(name.lower())
        banned = path.suffix in BANNED or bool(set(path.parts) & BLOCK_DIRS)
        banned |= path.suffix in {".csv", ".tsv"} and name not in ALLOWED_CSV
        banned |= path.name.startswith(".env") and path.name != ".env.example"
        banned |= (
            path.name.endswith((".local.yaml", ".local.json")) or name == "configs/data/local.yaml"
        )
        banned |= any(
            x in path.name for x in ("credential", "secret", "token")
        ) and path.suffix in {".json", ".yaml", ".toml"}
        banned |= name == "registry/checksums.sha256" or path.name == ".rhistory"
        if banned:
            errors.append(f"Private/raw path forbidden: {name}")
        blob = subprocess.run(
            ["git", "cat-file", "-s", f":{name}"], capture_output=True, text=True, check=False
        )
        if blob.returncode == 0 and int(blob.stdout) > 2 * 1024**2:
            errors.append(f"Tracked blob exceeds 2 MiB: {name}")
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
