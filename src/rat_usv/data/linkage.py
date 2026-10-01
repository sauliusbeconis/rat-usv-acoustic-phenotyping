"""Load and check the frozen recording linkage (WAV -> animal, date, age, litter, dam, weight)."""

import csv
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ANIMAL_ID = re.compile(r"^E\d+L\d+-\d{4}$")
LITTER_ID = re.compile(r"^E\d+L\d+$")
DAM_ID = re.compile(r"^E\d+D\d+$")
MARKING = re.compile(r"^\d\.\d\.\d\.\d$")


def _none(value: Any) -> Any:
    return None if value == "" else value


class Recording(BaseModel):
    """One source WAV. Pup fields are null for non-pup recordings (ambient, dam)."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    wav_path: str
    recording_date: date | None
    date_folder: str
    group_folder: str
    animal_id: str | None
    marking: str | None
    sex: str | None
    litter_id: str | None
    dam_id: str | None
    birth_date: date | None
    postnatal_age_days: int | None
    weight_g: float | None
    weight_scan: str | None
    weight_scan_row: str | None
    in_final_record_list: bool
    exclusion_reason: str | None
    wav_duration_s: float
    edit_suspected: bool
    wav_size_bytes: int
    identity_basis: str

    @field_validator("*", mode="before")
    @classmethod
    def empty_is_null(cls, value: Any) -> Any:
        return _none(value)

    @field_validator("wav_path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"wav_path must be relative without '..': {value}")
        return value

    @model_validator(mode="after")
    def consistent(self) -> "Recording":
        if self.animal_id is None:
            return self
        if not ANIMAL_ID.match(self.animal_id):
            raise ValueError(f"bad animal_id {self.animal_id}")
        if self.marking is None or not MARKING.match(self.marking):
            raise ValueError(f"{self.animal_id}: marking must be X.X.X.X, got {self.marking}")
        if self.animal_id.split("-")[1] != self.marking.replace(".", ""):
            raise ValueError(f"{self.animal_id} does not encode marking {self.marking}")
        if self.litter_id is None or not LITTER_ID.match(self.litter_id):
            raise ValueError(f"{self.animal_id}: bad litter_id {self.litter_id}")
        if not self.animal_id.startswith(self.litter_id + "-"):
            raise ValueError(f"{self.animal_id} not in litter {self.litter_id}")
        if self.dam_id is not None and not DAM_ID.match(self.dam_id):
            raise ValueError(f"{self.animal_id}: bad dam_id {self.dam_id}")
        if self.sex not in {"female", "male"}:
            raise ValueError(f"{self.animal_id}: sex must be female/male, got {self.sex}")
        if self.recording_date is None or self.birth_date is None:
            raise ValueError(f"{self.animal_id}: pup recording needs recording and birth dates")
        if self.postnatal_age_days != (self.recording_date - self.birth_date).days:
            raise ValueError(f"{self.wav_path}: age is not recording_date - birth_date (P0)")
        return self

    @property
    def is_final_pup(self) -> bool:
        return self.in_final_record_list and self.animal_id is not None


def load_recordings(linkage_dir: Path) -> list[Recording]:
    """Parse and validate `recordings.csv`; raises on any malformed row or duplicate path."""
    with (linkage_dir / "recordings.csv").open(newline="") as handle:
        rows = [Recording.model_validate(row) for row in csv.DictReader(handle)]
    duplicates = [p for p, n in Counter(r.wav_path for r in rows).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate wav_path rows: {duplicates[:5]}")
    keys = Counter((r.animal_id, r.recording_date) for r in rows if r.is_final_pup)
    repeated = [k for k, n in keys.items() if n > 1]
    if repeated:
        raise ValueError(f"more than one final recording per animal-day: {repeated[:5]}")
    return rows


def final_pup_recordings(rows: list[Recording]) -> list[Recording]:
    return [r for r in rows if r.is_final_pup]


def summarize(rows: list[Recording]) -> dict[str, Any]:
    final = final_pup_recordings(rows)
    ages: dict[str, set[int]] = {}
    for r in final:
        assert r.litter_id is not None and r.postnatal_age_days is not None
        ages.setdefault(r.litter_id, set()).add(r.postnatal_age_days)
    return {
        "recordings": len(rows),
        "final_pup_recordings": len(final),
        "animals": len({r.animal_id for r in final}),
        "litters": len(ages),
        "dams": len({r.dam_id for r in final if r.dam_id}),
        "sex_animals": dict(Counter(s for _, s in {(r.animal_id, r.sex) for r in final})),
        "recordings_per_litter": dict(sorted(Counter(r.litter_id for r in final).items())),
        "age_coverage": {
            litter: {
                "min": min(a),
                "max": max(a),
                "missing": sorted(set(range(min(a), max(a) + 1)) - a),
            }
            for litter, a in sorted(ages.items())
        },
        "edit_suspected_final": sum(r.edit_suspected for r in final),
        "final_without_weight": sum(r.weight_g is None for r in final),
    }


def check_sources(rows: list[Recording], source_root: Path) -> list[dict[str, Any]]:
    """Stat each WAV (no reads); report missing files or size changes."""
    problems: list[dict[str, Any]] = []
    for r in rows:
        path = source_root / r.wav_path
        if not path.is_file():
            problems.append({"wav_path": r.wav_path, "problem": "missing"})
        elif path.stat().st_size != r.wav_size_bytes:
            problems.append(
                {
                    "wav_path": r.wav_path,
                    "problem": "size_changed",
                    "expected": r.wav_size_bytes,
                    "found": path.stat().st_size,
                }
            )
    return problems
