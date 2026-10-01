"""Transparent USV detector: whitened spectrogram, tonal peak threshold, click rejection."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage, signal

FloatArray = NDArray[np.float32]


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nperseg: int = Field(default=512, gt=0)
    hop: int = Field(default=128, gt=0)
    fmin_hz: float = 18_000
    fmax_hz: float = 100_000
    smooth_freq_bins: int = Field(default=3, ge=1)
    smooth_frames: int = Field(default=5, ge=1)
    chunk_s: float = Field(default=30.0, gt=1)
    chunk_overlap_s: float = Field(default=0.5, ge=0)
    high_db: float = Field(default=10.0, description="whitened peak needed to seed a call")
    low_db: float = Field(default=6.0, description="whitened peak that extends a call")
    click_fraction: float = Field(
        default=0.3, gt=0, le=1, description="frames with this share of band bins above low_db"
    )
    max_peak_width_khz: float = Field(
        default=4.0, gt=0, description="tonality: band width within 10 dB of the frame peak"
    )
    min_peak_freq_khz: float = Field(
        default=25.0, description="reject calls whose loudest frame is below this (band-edge noise)"
    )
    merge_gap_ms: float = 8.0
    min_duration_ms: float = 4.0
    max_duration_ms: float = 1500.0


def load_detector_config(path: Path) -> DetectorConfig:
    return DetectorConfig.model_validate(yaml.safe_load(path.read_text()) or {})


@dataclass(frozen=True)
class Call:
    start_s: float
    end_s: float
    duration_ms: float
    peak_freq_khz: float
    min_freq_khz: float
    max_freq_khz: float
    mean_freq_khz: float
    start_freq_khz: float
    end_freq_khz: float
    peak_snr_db: float
    mean_snr_db: float
    rejected_frame_share: float

    def row(self) -> dict[str, Any]:
        return asdict(self)


def whitened_spectrogram(
    x: FloatArray, fs: int, cfg: DetectorConfig
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Band-limited, locally smoothed power (dB) minus each frequency bin's median over time."""
    f, t, power = signal.spectrogram(
        x, fs, window="hann", nperseg=cfg.nperseg, noverlap=cfg.nperseg - cfg.hop, mode="psd"
    )
    band = (f >= cfg.fmin_hz) & (f <= min(cfg.fmax_hz, fs / 2))
    smoothed = ndimage.uniform_filter(power[band], size=(cfg.smooth_freq_bins, cfg.smooth_frames))
    db = (10 * np.log10(smoothed + 1e-20)).astype(np.float32)
    db -= np.median(db, axis=1, keepdims=True)
    return f[band].astype(np.float32), t.astype(np.float32), db


def _segments(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def detect_array(x: FloatArray, fs: int, cfg: DetectorConfig, offset_s: float = 0.0) -> list[Call]:
    f, t, db = whitened_spectrogram(x, fs, cfg)
    if db.shape[1] == 0:
        return []
    frame_s = cfg.hop / fs
    peak = db.max(axis=0)
    peak_bin = db.argmax(axis=0)
    above = db > cfg.low_db
    click = above.mean(axis=0) >= cfg.click_fraction
    bin_khz = float(f[1] - f[0]) / 1000 if f.size > 1 else 1.0
    width_khz = (above & (db > peak - 10)).sum(axis=0) * bin_khz
    reject = click | (width_khz > cfg.max_peak_width_khz)
    active = (peak > cfg.low_db) & ~reject
    # Merge short gaps, then keep segments that reach the seed threshold.
    gap = max(1, round(cfg.merge_gap_ms / 1000 / frame_s))
    for a, b in _segments(~active):
        if 0 < a and b < active.size and b - a <= gap:
            active[a:b] = True
    calls = []
    for a, b in _segments(active):
        frames = np.arange(a, b)
        good = frames[~reject[frames] & (peak[frames] > cfg.low_db)]
        if good.size == 0 or peak[good].max() < cfg.high_db:
            continue
        start = float(t[a]) - frame_s / 2 + offset_s
        end = float(t[b - 1]) + frame_s / 2 + offset_s
        duration_ms = (end - start) * 1000
        if not cfg.min_duration_ms <= duration_ms <= cfg.max_duration_ms:
            continue
        contour = f[peak_bin[good]] / 1000
        best = good[peak[good].argmax()]
        if f[peak_bin[best]] / 1000 < cfg.min_peak_freq_khz:
            continue
        calls.append(
            Call(
                start_s=round(start, 5),
                end_s=round(end, 5),
                duration_ms=round(duration_ms, 3),
                peak_freq_khz=round(float(f[peak_bin[best]]) / 1000, 3),
                min_freq_khz=round(float(contour.min()), 3),
                max_freq_khz=round(float(contour.max()), 3),
                mean_freq_khz=round(float(contour.mean()), 3),
                start_freq_khz=round(float(contour[0]), 3),
                end_freq_khz=round(float(contour[-1]), 3),
                peak_snr_db=round(float(peak[good].max()), 2),
                mean_snr_db=round(float(peak[good].mean()), 2),
                rejected_frame_share=round(float(reject[frames].mean()), 3),
            )
        )
    return calls


def detect_file(path: Path, cfg: DetectorConfig) -> tuple[list[Call], float]:
    """Detect over a WAV in overlapping chunks (read-only); returns calls and duration (s)."""
    info = sf.info(str(path))
    fs, total = info.samplerate, info.frames
    step = int(cfg.chunk_s * fs)
    pad = int(cfg.chunk_overlap_s * fs)
    calls: list[Call] = []
    with sf.SoundFile(str(path), mode="r") as handle:
        for begin in range(0, total, step):
            lo = max(0, begin - pad)
            handle.seek(lo)
            x = handle.read(min(total, begin + step + pad) - lo, dtype="float32", always_2d=True)
            found = detect_array(x[:, 0], fs, cfg, offset_s=lo / fs)
            # Keep a call in the chunk that owns its start to avoid duplicates across overlaps.
            calls.extend(c for c in found if begin / fs <= c.start_s < (begin + step) / fs)
    return calls, total / fs
