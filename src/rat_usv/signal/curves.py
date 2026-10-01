"""M9 work package 0: per-call time-normalised curves and nonlinear-phenomena flags.

Curves (decision 0020) are sampled at N_POINTS over each series' own frame times:
- f0: the M2 fundamental contour (kHz), after trimming and octave correction;
- envelope: frame peak level (whitened dB, clipped at 0) over *all* call frames, so weak
  onsets and tails are kept;
- harmonic ratio: level at 2*f0 minus level at f0 (dB), NaN where 2*f0 is out of band;
- tonality: 1 - normalised spectral entropy of the frame.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rat_usv.signal.detect import DetectorConfig

N_POINTS = 32
SUB_TOL_KHZ = 1.0  # subharmonic window around f0/2
SUB_LOCAL_KHZ = 5.0  # local median window around f0/2 (excluding the +/- SUB_TOL window)
SUB_MIN_DB = 6.0
BIPHON_EXCLUDE_KHZ = 2.0  # a second band must not lie within this of k*f0, k in {0.5, 1, 2}
BIPHON_CONTINUITY_KHZ = 2.0  # frame-to-frame frequency change allowed within one band
MIN_RUN = 3  # frames


def resample(times: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float32]:
    """Linear interpolation of a series onto N_POINTS over its own (finite) time span."""
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.full(N_POINTS, np.nan, dtype=np.float32)
    t, v = times[ok], values[ok]
    tau = (t - times[0]) / max(float(times[-1] - times[0]), 1e-9)
    return np.interp(np.linspace(0, 1, N_POINTS), tau, v).astype(np.float32)


def _window(col: NDArray[np.float32], f_khz: NDArray[np.float32], lo: float, hi: float) -> Any:
    return col[(f_khz >= lo) & (f_khz <= hi)]


def _peak(col: NDArray[np.float32], f_khz: NDArray[np.float32], lo: float, hi: float) -> float:
    """Highest level in [lo, hi] kHz; NaN if the window lies outside the band."""
    w = _window(col, f_khz, lo, hi)
    return float(w.max()) if w.size else np.nan


def runs(flags: NDArray[np.bool_], min_run: int = MIN_RUN) -> int:
    """Number of runs of at least `min_run` consecutive True values."""
    n, length = 0, 0
    for f in [*flags.tolist(), False]:
        if f:
            length += 1
        else:
            n += length >= min_run
            length = 0
    return n


def subharmonic_flags(
    db: NDArray[np.float32], f_khz: NDArray[np.float32], f0: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Per kept frame: energy at f0/2 (+/- 1 kHz) >= 6 dB above the local median."""
    out = np.zeros(f0.size, dtype=bool)
    for i, fk in enumerate(f0):
        half = fk / 2
        if half - SUB_LOCAL_KHZ < float(f_khz[0]):
            continue
        col = db[:, i]
        peak = _window(col, f_khz, half - SUB_TOL_KHZ, half + SUB_TOL_KHZ)
        near = (np.abs(f_khz - half) <= SUB_LOCAL_KHZ) & (np.abs(f_khz - half) > SUB_TOL_KHZ)
        if peak.size and near.any():
            out[i] = float(peak.max()) - float(np.median(col[near])) >= SUB_MIN_DB
    return out


def biphonation_flags(
    db: NDArray[np.float32],
    f_khz: NDArray[np.float32],
    f0: NDArray[np.float64],
    cfg: DetectorConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Per kept frame: a narrow second band (>= high_db) away from 0.5, 1 and 2 x f0."""
    bin_khz = float(f_khz[1] - f_khz[0])
    flag = np.zeros(f0.size, dtype=bool)
    freq = np.full(f0.size, np.nan)
    for i, fk in enumerate(f0):
        col = db[:, i].copy()
        # Exclude the whole above-threshold lobe around every multiple of f0/2 (decision 0021).
        mask = np.zeros(col.size, dtype=bool)
        for k in np.arange(0.5, float(f_khz[-1]) / fk + 0.5, 0.5):
            mask |= np.abs(f_khz - k * fk) <= BIPHON_EXCLUDE_KHZ
        above = col > cfg.low_db
        grow = mask.copy()
        while True:
            nxt = grow | (np.r_[False, grow[:-1]] & above) | (np.r_[grow[1:], False] & above)
            if (nxt == grow).all():
                break
            grow = nxt
        col[grow] = -np.inf
        j = int(np.argmax(col))
        peak = float(col[j])
        if peak < cfg.high_db:
            continue
        width = float(((col > cfg.low_db) & (col > peak - 10)).sum() * bin_khz)
        if width <= cfg.max_peak_width_khz * 2:
            flag[i], freq[i] = True, float(f_khz[j])
    # A band must be continuous in frequency across consecutive frames.
    cont = flag.copy()
    for i in range(1, f0.size):
        if flag[i] and flag[i - 1] and abs(freq[i] - freq[i - 1]) > BIPHON_CONTINUITY_KHZ:
            cont[i] = False
    return cont, freq


def call_curves(
    f_khz: NDArray[np.float32],
    t: NDArray[np.float32],
    db: NDArray[np.float32],
    times: NDArray[np.float64],
    contour: NDArray[np.float64],
    columns: NDArray[np.int64],
    cfg: DetectorConfig,
) -> dict[str, Any]:
    """Curves and nonlinear-phenomena counts for one call (db: bins x all call frames)."""
    kept = db[:, columns]
    f0_db = np.array([_peak(kept[:, i], f_khz, f - 1, f + 1) for i, f in enumerate(contour)])
    harm = np.full(contour.size, np.nan)
    for i, f in enumerate(contour):
        if 2 * f + 2.0 <= float(f_khz[-1]):
            harm[i] = _peak(kept[:, i], f_khz, 2 * f - 2, 2 * f + 2) - f0_db[i]
    lin = np.power(10.0, np.clip(kept, 0, None) / 10) - 1
    p = lin / np.where(lin.sum(0) > 0, lin.sum(0), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0) / np.log(p.shape[0])
    envelope = np.clip(db.max(axis=0), 0, None).astype(np.float64)
    sub = subharmonic_flags(kept, f_khz, contour)
    bip, _ = biphonation_flags(kept, f_khz, contour, cfg)
    return {
        "f0": resample(times, contour),
        "envelope": resample(t.astype(np.float64), envelope),
        "harmonic_ratio": resample(times, harm),
        "tonality": resample(times, 1 - ent),
        "raw_t": times.astype(np.float32),
        "raw_f0": contour.astype(np.float32),
        "raw_env_t": t.astype(np.float32),
        "raw_env": envelope.astype(np.float32),
        "n_subharmonic_events": runs(sub),
        "subharmonic_frame_share": float(sub.mean()),
        "n_biphonation_events": runs(bip),
        "biphonation_frame_share": float(bip.mean()),
    }
