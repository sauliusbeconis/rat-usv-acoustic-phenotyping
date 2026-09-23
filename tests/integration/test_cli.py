import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from rat_usv.cli import app
from rat_usv.schemas.registry import initialize_empty


def test_full_synthetic_cli_pipeline(source: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "local.yaml"
    cfg.write_text(
        f"source_root: {source}\noutput_root: {tmp_path / 'runs'}\nhash:\n  mode: full\naudit_hash:\n  mode: full\n"
    )
    runner = CliRunner()

    def run(*args: str) -> Path:
        result = runner.invoke(app, [*args, "--config", str(cfg)])
        assert result.exit_code == 0, result.output + str(result.exception)
        path = Path(result.stdout.strip())
        assert (path / "run.json").is_file()
        return path

    inv = run("inventory")
    aud = run("audit", "--inventory", str(inv / "source_inventory.parquet"))
    run("crosswalk", "--inventory", str(inv / "source_inventory.parquet"))
    atlas = run(
        "dataset-atlas",
        "--inventory",
        str(inv / "source_inventory.parquet"),
        "--audit",
        str(aud / "wav_audit.parquet"),
    )
    data = json.loads((atlas / "atlas.json").read_text())
    assert data["source_files"] == 3
    assert data["biological_counts"]["animals"] is None
    registry = tmp_path / "registry"
    initialize_empty(registry)
    val = run(
        "validate",
        "--registry",
        str(registry),
        "--source-inventory",
        str(inv / "source_inventory.parquet"),
        "--verify-hashes",
    )
    assert json.loads((val / "validation.json").read_text())["source_hashes"]["checked"] == 3
    (source / "mapping.csv").write_text("modified")
    failure = runner.invoke(
        app,
        [
            "validate",
            "--config",
            str(cfg),
            "--registry",
            str(registry),
            "--source-inventory",
            str(inv / "source_inventory.parquet"),
            "--verify-hashes",
        ],
    )
    assert failure.exit_code == 1


def test_only_implemented_commands_exposed() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("inventory", "audit", "validate", "crosswalk", "dataset-atlas"):
        assert command in result.output
    assert "train " not in result.output


def test_git_guard_blocks_forced_private_index(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/check_git_safety.py"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    private_names = (
        "private.WAV",
        "sheet.XLSX",
        ".codex/session.json",
        "docs/.agents/session.md",
        "Supplments/image.JPG",
        "Supplments/notes.txt",
        "Supplements/notes.txt",
    )
    ignore = script.parents[1] / ".gitignore"
    (tmp_path / ".gitignore").write_text(ignore.read_text())
    for name in (*private_names, "safe.md"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic fixture only")
    ignored = subprocess.run(
        ["git", "check-ignore", "--", *private_names],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(ignored.stdout.splitlines()) == set(private_names)
    subprocess.run(["git", "add", "-f", "--", *private_names, "safe.md"], cwd=tmp_path, check=True)
    result = subprocess.run(
        [sys.executable, str(script), "--staged"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert all(name in result.stdout for name in private_names)
    assert "safe.md" not in result.stdout
