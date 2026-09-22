from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rat_usv.config import HashPolicy, SourceConfig


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    for directory in ("session one", "session two"):
        (root / directory).mkdir()
    x = np.zeros((1000, 2))
    x[:10, 0] = 0.99999
    x[10:20, 0] = -1
    sf.write(root / "session one" / "0001.WAV", x, 250000, subtype="PCM_16")
    sf.write(root / "session two" / "0001.wav", np.zeros(700), 192000, subtype="PCM_24")
    (root / "mapping.csv").write_text("alias,day\n0001,unknown\n")
    return root


@pytest.fixture
def cfg(source: Path, tmp_path: Path) -> SourceConfig:
    return SourceConfig(
        source_root=source,
        output_root=tmp_path / "runs",
        hash=HashPolicy(mode="full"),
        audit_hash=HashPolicy(mode="full"),
    )


@pytest.fixture
def split_rows() -> list[dict[str, str]]:
    return [
        {
            "recording_id": "r1",
            "animal_id": "a1",
            "litter_id": "l1",
            "audio_sha256": "1" * 64,
            "split": "train",
        },
        {
            "recording_id": "r2",
            "animal_id": "a2",
            "litter_id": "l2",
            "audio_sha256": "2" * 64,
            "split": "validation",
        },
        {
            "recording_id": "r3",
            "animal_id": "a3",
            "litter_id": "l3",
            "audio_sha256": "3" * 64,
            "split": "test",
        },
    ]
