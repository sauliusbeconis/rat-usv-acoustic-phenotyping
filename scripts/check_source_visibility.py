"""Check Git-visible delivery coverage without staging or changing the real index."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    visible = set(
        filter(
            None,
            subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repo
            )
            .decode()
            .split("\0"),
        )
    )
    required = {str(p.relative_to(repo)) for p in (repo / "src").rglob("*.py")}
    required |= {"configs/data/example.yaml", "pyproject.toml", "uv.lock"}
    hidden = sorted(required - visible)
    if hidden:
        print(json.dumps({"hidden_required_files": hidden}))
        return 1
    with tempfile.TemporaryDirectory(prefix="usv-checkout-") as temp:
        root = Path(temp)
        candidate, checkout = root / "candidate", root / "checkout"
        candidate.mkdir()
        for name in sorted(visible):
            source = repo / name
            if source.is_file():
                destination = candidate / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        # Construct a candidate index in a disposable repository, export that index,
        # then verify imports resolve from the exported tree, never the editable install.
        subprocess.run(["git", "init", "-q", str(candidate)], check=True)
        subprocess.run(["git", "add", "."], cwd=candidate, check=True)
        subprocess.run(
            [sys.executable, str(candidate / "scripts/check_git_safety.py"), "--staged"],
            cwd=candidate,
            check=True,
        )
        subprocess.run(
            ["git", "checkout-index", "--all", f"--prefix={checkout}/"], cwd=candidate, check=True
        )
        code = """
import importlib, json, pathlib, pkgutil, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / 'src'))
import rat_usv
modules = [rat_usv] + [importlib.import_module(m.name) for m in pkgutil.walk_packages(rat_usv.__path__, 'rat_usv.')]
assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in modules)
assert (root / 'configs/data/example.yaml').is_file()
assert not (root / 'configs/data/local.yaml').exists()
print(json.dumps({'imported_modules': len(modules), 'candidate_clean_checkout': 'passed'}))
"""
        subprocess.run([sys.executable, "-I", "-c", code, str(checkout)], cwd=checkout, check=True)
    print(json.dumps({"required_files_visible": len(required), "actual_index_modified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
