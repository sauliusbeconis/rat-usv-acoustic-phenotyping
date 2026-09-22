"""Validated, explicit local paths and bounded read budgets."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HashPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "budget", "full"] = "budget"
    max_file_bytes: int = Field(default=64 * 1024**2, ge=0)
    max_total_bytes: int = Field(default=512 * 1024**2, ge=0)


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_root: Path
    output_root: Path
    exclude_relative: list[str] = Field(default_factory=list)
    hash: HashPolicy = Field(default_factory=HashPolicy)
    audit_hash: HashPolicy = Field(default_factory=HashPolicy)
    audio_window_seconds: float = Field(default=1.0, gt=0, le=10)

    @model_validator(mode="after")
    def safe_paths(self) -> "SourceConfig":
        self.source_root = self.source_root.resolve(strict=True)
        self.output_root = self.output_root.resolve()
        if not self.source_root.is_dir():
            raise ValueError("source_root must be a directory")
        for item in self.exclude_relative:
            rel = Path(item)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError("Exclusions must be nonempty relative paths without '..'")
            excluded = (self.source_root / rel).resolve()
            if excluded == self.source_root or not excluded.is_relative_to(self.source_root):
                raise ValueError("Exclusions must stay strictly below source_root")
        if self.output_root.is_relative_to(self.source_root):
            if not any(
                self.output_root.is_relative_to((self.source_root / x).resolve())
                for x in self.exclude_relative
            ):
                raise ValueError(
                    "output_root inside source must be under an explicit excluded subtree"
                )
        return self


def load_config(path: Path) -> SourceConfig:
    path = path.resolve(strict=True)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("source_root", "output_root"):
        if key in raw:
            value = Path(raw[key]).expanduser()
            raw[key] = value if value.is_absolute() else path.parent / value
    return SourceConfig.model_validate(raw)
