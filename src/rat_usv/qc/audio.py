"""WAV header and bounded amplitude diagnostics; never rewrites source audio."""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from rat_usv.config import SourceConfig
from rat_usv.provenance.io import Hasher, record_run, safe_source, write_json

AUDIO_SCHEMA = pa.schema(
    [
        ("relative_path", pa.string()),
        ("size_bytes", pa.int64()),
        ("sample_rate_hz", pa.int64()),
        ("channels", pa.int64()),
        ("subtype", pa.string()),
        ("bit_depth", pa.int64()),
        ("sample_width_bytes", pa.int64()),
        ("frames", pa.int64()),
        ("duration_seconds", pa.float64()),
        ("readability", pa.string()),
        ("sha256", pa.string()),
        ("hash_status", pa.string()),
        ("diagnostics_json", pa.string()),
        ("notes", pa.string()),
    ]
)
BITS = {
    "PCM_U8": 8,
    "PCM_S8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


def inspect_wav(path: Path, window_seconds: float) -> dict[str, Any]:
    """Decode up to three disjoint windows; report coverage, not whole-file readability."""
    import json

    row: dict[str, Any] = {
        "sample_rate_hz": None,
        "channels": None,
        "subtype": None,
        "bit_depth": None,
        "sample_width_bytes": None,
        "frames": None,
        "duration_seconds": None,
        "readability": "unreadable",
        "diagnostics_json": None,
        "notes": "",
    }
    try:
        with sf.SoundFile(path, mode="r") as audio:
            bits = BITS.get(audio.subtype)
            row.update(
                sample_rate_hz=audio.samplerate,
                channels=audio.channels,
                subtype=audio.subtype,
                bit_depth=bits,
                sample_width_bytes=(bits // 8 if bits else None),
                frames=audio.frames,
                duration_seconds=audio.frames / audio.samplerate,
            )
            n = min(audio.frames, max(1, int(window_seconds * audio.samplerate)))
            starts = sorted({0, max(0, (audio.frames - n) // 2), max(0, audio.frames - n)})
            windows = []
            decoded = 0
            end = 0
            for start in starts:
                start = max(start, end)
                stop = min(audio.frames, start + n)
                if stop <= start:
                    continue
                audio.seek(start)
                data = audio.read(stop - start, dtype="float64", always_2d=True)
                if len(data) != stop - start:
                    raise ValueError("Short audio read")
                decoded += len(data)
                end = stop
                finite = np.isfinite(data)
                threshold = (
                    1.0 - 2.0 ** (-(bits - 1)) if bits and audio.subtype.startswith("PCM") else 1.0
                )
                channels = []
                for c in range(audio.channels):
                    x = data[:, c][finite[:, c]]
                    channels.append(
                        {
                            "channel_index": c,
                            "peak_absolute": float(np.max(np.abs(x))) if x.size else None,
                            "rms": float(np.sqrt(np.mean(x**2))) if x.size else None,
                            "mean_dc": float(np.mean(x)) if x.size else None,
                            "full_scale_fraction": float(np.mean((x <= -1) | (x >= threshold)))
                            if x.size
                            else None,
                            "digital_zero_fraction": float(np.mean(x == 0)) if x.size else None,
                            "nonfinite_samples": int((~finite[:, c]).sum()),
                        }
                    )
                windows.append({"start_frame": start, "stop_frame": stop, "channels": channels})
            row.update(
                readability="empty" if not audio.frames else "sampled_decode_ok",
                diagnostics_json=json.dumps(
                    {
                        "decoded_frames": decoded,
                        "coverage_fraction": decoded / audio.frames if audio.frames else 0,
                        "windows": windows,
                    },
                    allow_nan=False,
                ),
                notes="Header and sampled windows only. Full-scale samples are a clipping proxy, not an exclusion decision. Header frame count may reflect decoder handling of truncated files.",
            )
    except (OSError, RuntimeError, ValueError) as e:
        row.update(readability="unreadable_or_decode_error", notes=str(e))
    return row


def audit(config: SourceConfig, inventory_path: Path, out: Path) -> list[dict[str, Any]]:
    items = pq.read_table(inventory_path).to_pylist()
    hasher = Hasher(config.audit_hash)
    rows = []
    for item in sorted(items, key=lambda x: x["relative_path"]):
        if item["file_type"] != ".wav":
            continue
        row: dict[str, Any] = {
            "relative_path": item["relative_path"],
            "size_bytes": None,
            "sha256": None,
            "hash_status": "not_attempted",
        }
        try:
            path = safe_source(config.source_root, item["relative_path"])
            before = path.stat()
            row["size_bytes"] = before.st_size
            row.update(inspect_wav(path, config.audio_window_seconds))
            row["sha256"], row["hash_status"] = hasher.compute(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                row.update(
                    readability="changed_during_audit",
                    sha256=None,
                    hash_status="changed_during_read",
                )
            if item.get("sha256") and row.get("sha256") and item["sha256"] != row["sha256"]:
                row.update(
                    readability="hash_mismatch", notes="Source hash differs from supplied inventory"
                )
        except (OSError, ValueError) as e:
            row.update(readability="path_error", notes=str(e))
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=AUDIO_SCHEMA), out / "wav_audit.parquet")
    write_json(
        out / "summary.json",
        {
            "wav_files": len(rows),
            "readability": dict(Counter(r["readability"] for r in rows)),
            "hash_status": dict(Counter(r["hash_status"] for r in rows)),
            "hash_bytes_read": hasher.bytes_read,
            "total_header_duration_seconds": sum(r.get("duration_seconds") or 0 for r in rows),
            "sample_rates": dict(Counter(str(r.get("sample_rate_hz")) for r in rows)),
            "full_decode_verified": False,
        },
    )
    record_run(out, "audit", config, [inventory_path])
    return rows
