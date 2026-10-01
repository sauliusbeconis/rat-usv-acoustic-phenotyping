"""Spectral summaries of call-free background windows (negative-control input)."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

BAND_EDGES_KHZ = np.linspace(0, 105, 15)  # 14 bands of 7.5 kHz up to just below Nyquist


def window_bands(x: np.ndarray, fs: int) -> np.ndarray:
    f, p = signal.welch(x, fs, nperseg=1024)
    out = []
    for lo, hi in zip(BAND_EDGES_KHZ[:-1], BAND_EDGES_KHZ[1:], strict=True):
        sel = (f >= lo * 1000) & (f < hi * 1000)
        out.append(10 * np.log10(p[sel].mean() + 1e-20))
    return np.array(out)


def _recording(args: tuple[Path, list[tuple[float, float]]]) -> np.ndarray:
    path, windows = args
    fs = sf.info(str(path)).samplerate
    rows = []
    for a, b in windows:
        x, _ = sf.read(str(path), start=int(a * fs), stop=int(b * fs), dtype="float32")
        rows.append(window_bands(x, fs))
    return np.mean(rows, axis=0) if rows else np.full(len(BAND_EDGES_KHZ) - 1, np.nan)


def background_features(
    source_root: Path, windows: list[dict[str, Any]], workers: int
) -> dict[str, list[float]]:
    """wav_path -> mean band power (dB) over its background windows."""
    by_rec: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for w in windows:
        by_rec[w["wav_path"]].append((w["start_s"], w["end_s"]))
    paths = sorted(by_rec)
    with ProcessPoolExecutor(workers) as pool:
        values = list(pool.map(_recording, [(source_root / p, by_rec[p]) for p in paths]))
    return {p: [float(v) for v in vals] for p, vals in zip(paths, values, strict=True)}


def band_names() -> list[str]:
    return [
        f"bg_{lo:.1f}_{hi:.1f}khz"
        for lo, hi in zip(BAND_EDGES_KHZ[:-1], BAND_EDGES_KHZ[1:], strict=True)
    ]
