"""Animal- and litter-level splits, written once with a content digest."""

import hashlib
import json
from pathlib import Path
from typing import Any


def make_splits(animals: dict[str, str]) -> dict[str, Any]:
    """animals: animal_id -> litter_id. Leave-one-animal-out (primary) and leave-one-litter-out."""
    ids = sorted(animals)
    litters = sorted(set(animals.values()))
    return {
        "schema": "rat-usv-splits-v1",
        "unit": "animal",
        "animals": {a: animals[a] for a in ids},
        "leave_one_animal_out": [{"fold": i, "test_animals": [a]} for i, a in enumerate(ids)],
        "leave_one_litter_out": [
            {"fold": i, "test_animals": [a for a in ids if animals[a] == litter], "litter": litter}
            for i, litter in enumerate(litters)
        ],
    }


def write_splits(splits: dict[str, Any], path: Path) -> str:
    data = json.dumps(splits, indent=1, sort_keys=True).encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def load_splits(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Split file digest mismatch: {path}")
    splits: dict[str, Any] = json.loads(data)
    for name in ("leave_one_animal_out", "leave_one_litter_out"):
        seen = [a for fold in splits[name] for a in fold["test_animals"]]
        if sorted(seen) != sorted(splits["animals"]) or len(seen) != len(set(seen)):
            raise ValueError(f"{name}: every animal must be tested exactly once")
    return splits
