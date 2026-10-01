"""Fixed-size call spectrogram images (AVA-style) for representation learning.

Each call is cut from the detector's whitened spectrogram (dB above the bin's median),
clipped to [0, CLIP_DB], resampled to SIZE x SIZE. Time uses AVA's square-root stretch:
a call of duration d is shown over a window of sqrt(d * MAX_DUR) seconds centred on the call,
so short calls are not lost and long calls fit. No statistic is fitted to the data.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from numpy.typing import NDArray
from scipy import interpolate

from rat_usv.signal.detect import DetectorConfig, whitened_spectrogram

SIZE = 128
CLIP_DB = 40.0
MAX_DUR_S = 0.25
BACKGROUND_CROP_S = 0.05


def to_image(
    f_khz: NDArray[np.float32],
    t: NDArray[np.float32],
    db: NDArray[np.float32],
    centre: float,
    dur: float,
) -> NDArray[np.uint8]:
    span = float(np.sqrt(min(dur, MAX_DUR_S) * MAX_DUR_S))
    span = max(span, min(dur, MAX_DUR_S))
    grid_t = np.linspace(centre - span / 2, centre + span / 2, SIZE)
    grid_f = np.linspace(float(f_khz[0]), float(f_khz[-1]), SIZE)
    clipped = np.clip(db, 0, CLIP_DB) / CLIP_DB
    fn = interpolate.RegularGridInterpolator(
        (f_khz, t), clipped, bounds_error=False, fill_value=0.0
    )
    ff, tt = np.meshgrid(grid_f, grid_t, indexing="ij")
    img = fn(np.stack([ff.ravel(), tt.ravel()], axis=1)).reshape(SIZE, SIZE)
    return np.round(img[::-1] * 255).astype(np.uint8)  # high frequency at the top


def recording_images(
    path: Path, items: list[dict[str, Any]], cfg: DetectorConfig
) -> list[tuple[dict[str, Any], NDArray[np.uint8]]]:
    """items: dicts with start_s/end_s (+ any keys to carry). Chunked like the detector."""
    info = sf.info(str(path))
    fs, total = info.samplerate, info.frames
    step, pad = int(cfg.chunk_s * fs), int(cfg.chunk_overlap_s * fs)
    out = []
    with sf.SoundFile(str(path), mode="r") as handle:
        for begin in range(0, total, step):
            owned = [c for c in items if begin / fs <= c["start_s"] < (begin + step) / fs]
            if not owned:
                continue
            lo = max(0, begin - pad)
            handle.seek(lo)
            x = handle.read(min(total, begin + step + pad) - lo, dtype="float32", always_2d=True)
            f, t, db = whitened_spectrogram(x[:, 0], fs, cfg)
            t = t + lo / fs
            f_khz = (f / 1000).astype(np.float32)
            for c in owned:
                dur = c["end_s"] - c["start_s"]
                centre = c["start_s"] + dur / 2
                half = MAX_DUR_S / 2 + 0.01
                sel = (t >= centre - half) & (t <= centre + half)
                out.append((c, to_image(f_khz, t[sel], db[:, sel], centre, dur)))
    return out


def _job(
    args: tuple[Path, list[dict[str, Any]], DetectorConfig],
) -> list[tuple[dict[str, Any], NDArray[np.uint8]]]:
    return recording_images(*args)


KEEP = (
    "wav_path",
    "call_index",
    "animal_id",
    "litter_id",
    "postnatal_age_days",
    "start_s",
    "end_s",
)


def build_images(
    source_root: Path,
    detect_run: Path,
    features_run: Path,
    cfg: DetectorConfig,
    out: Path,
    workers: int,
) -> dict[str, Any]:
    calls = pq.read_table(features_run / "call_features.parquet").to_pylist()
    calls = [c for c in calls if c["confident"]]
    bg = pq.read_table(detect_run / "background_windows.parquet").to_pylist()
    for w in bg:  # a fixed-length crop from the middle of each call-free window
        mid = (w["start_s"] + w["end_s"]) / 2
        w["start_s"], w["end_s"] = mid - BACKGROUND_CROP_S / 2, mid + BACKGROUND_CROP_S / 2
    result: dict[str, Any] = {}
    for name, items in (("calls", calls), ("background", bg)):
        by_rec: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in items:
            by_rec[c["wav_path"]].append(
                {k: c.get(k) for k in KEEP} | {"window_index": c.get("window_index")}
            )
        jobs = [(source_root / p, cs, cfg) for p, cs in sorted(by_rec.items())]
        rows, images = [], []
        with ProcessPoolExecutor(workers) as pool:
            for pairs in pool.map(_job, jobs):
                for meta, img in pairs:
                    rows.append(meta)
                    images.append(img)
        np.save(out / f"{name}_images.npy", np.stack(images))
        pq.write_table(pa.Table.from_pylist(rows), out / f"{name}_index.parquet")
        result[name] = len(rows)
    return {"images": result, "size": SIZE, "clip_db": CLIP_DB, "max_dur_s": MAX_DUR_S,
            "background_crop_s": BACKGROUND_CROP_S, "call_set": "confident (peak SNR >= 14 dB)"}  # fmt: skip
