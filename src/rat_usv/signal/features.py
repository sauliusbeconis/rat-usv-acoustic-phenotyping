"""M2 per-call features on the fundamental-frequency contour, and per animal-day summaries."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from numpy.typing import NDArray
from scipy import ndimage

from rat_usv.config import SourceConfig
from rat_usv.provenance.io import write_json
from rat_usv.signal.detect import DetectorConfig, whitened_spectrogram

CONFIDENT_SNR_DB = 14.0  # decision 0008: primary call set
JUMP_KHZ = 10.0  # frequency jump: f0 change >= 10 kHz between adjacent frames
HALF_MIN_KHZ = 25.0  # a sub-peak at half frequency must be at least this to count as f0
HALF_WITHIN_DB = 15.0  # ... and no more than this below the loudest band
HARMONIC_TOL_KHZ = 2.0
WEAK_FRAME_DB = 10.0  # contour frames this far below the call's median power are dropped

FEATURES = (
    "duration_ms", "f0_start_khz", "f0_end_khz", "f0_min_khz", "f0_max_khz", "f0_mean_khz",
    "f0_median_khz", "f0_sd_khz", "bandwidth_khz", "slope_khz_per_s", "sinuosity",
    "n_jumps", "harmonic_share", "harmonic_observable_share", "tonality", "peak_freq_khz", "peak_snr_db", "mean_snr_db",
)  # fmt: skip


def _window_max(
    db: NDArray[np.float32], f_khz: NDArray[np.float32], centre: float, tol: float
) -> tuple[float, int]:
    sel = np.flatnonzero(np.abs(f_khz - centre) <= tol)
    if sel.size == 0:
        return -np.inf, -1
    k = sel[np.argmax(db[sel])]
    return float(db[k]), int(k)


def _persistent_jumps(contour: NDArray[np.float64], hold: int = 3) -> int:
    """Steps of >= JUMP_KHZ whose new level persists for `hold` frames on both sides."""
    n = 0
    for i in range(contour.size - 1):
        if abs(contour[i + 1] - contour[i]) < JUMP_KHZ:
            continue
        before, after = contour[max(0, i - hold + 1) : i + 1], contour[i + 1 : i + 1 + hold]
        if before.size >= hold and after.size >= hold:
            n += abs(float(np.median(after)) - float(np.median(before))) >= JUMP_KHZ
    return n


def call_features(
    f_khz: NDArray[np.float32], t: NDArray[np.float32], db: NDArray[np.float32], cfg: DetectorConfig
) -> dict[str, float | None] | None:
    """Features from the whitened spectrogram frames of one call (frames x bins in `db`)."""
    if db.shape[1] == 0:
        return None
    bin_khz = float(f_khz[1] - f_khz[0]) if f_khz.size > 1 else 1.0
    times, f0, power, harmonic, tonal = [], [], [], [], []
    for j in range(db.shape[1]):
        col = db[:, j]
        k = int(np.argmax(col))
        peak = float(col[k])
        width = float(((col > cfg.low_db) & (col > peak - 10)).sum() * bin_khz)
        if peak <= cfg.low_db or width > cfg.max_peak_width_khz * 2:
            continue
        fk = float(f_khz[k])
        half_db, half_k = _window_max(col, f_khz, fk / 2, HARMONIC_TOL_KHZ / 2)
        if fk / 2 >= HALF_MIN_KHZ and half_db > cfg.low_db and half_db >= peak - HALF_WITHIN_DB:
            fk = float(f_khz[half_k])
        # The harmonic is only observable if 2*f0 lies inside the analysed band (Nyquist limit).
        observable = 2 * fk + HARMONIC_TOL_KHZ <= float(f_khz[-1])
        harm_db, _ = (
            _window_max(col, f_khz, 2 * fk, HARMONIC_TOL_KHZ) if observable else (np.nan, -1)
        )
        lin = np.power(10.0, np.clip(col, 0, None) / 10) - 1
        p = lin / lin.sum() if lin.sum() > 0 else np.full(lin.size, 1 / lin.size)
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(p.size))
        times.append(float(t[j]))
        f0.append(fk)
        power.append(peak)
        harmonic.append(float(harm_db > cfg.low_db) if observable else np.nan)
        tonal.append(1 - entropy)
    if len(f0) < 2:
        return None
    keep = np.array(power) >= np.median(power) - WEAK_FRAME_DB  # drop weak tails/echo fragments
    if keep.sum() < 2:
        return None
    times = list(np.array(times)[keep])
    power = list(np.array(power)[keep])
    harmonic = list(np.array(harmonic, dtype=float)[keep])
    tonal = list(np.array(tonal)[keep])
    f0 = list(np.array(f0)[keep])
    ts = np.array(times)
    raw = np.array(f0)
    # Octave correction: frames that jumped to the harmonic (about 2x the call's median f0).
    centre = float(np.median(raw))
    ratio = raw / centre
    raw = np.where((ratio > 1.75) & (ratio < 2.25), raw / 2, raw)
    raw = np.where((ratio > 0.42) & (ratio < 0.58), raw * 2, raw)
    contour = ndimage.median_filter(raw, size=3, mode="nearest")
    pw = np.array(power)
    jumps = _persistent_jumps(contour)
    dt_ms = np.diff(ts) * 1000
    arc = float(np.sqrt(dt_ms**2 + np.diff(contour) ** 2).sum())
    n_edge = min(3, contour.size)
    start, end = float(np.median(contour[:n_edge])), float(np.median(contour[-n_edge:]))
    chord = float(np.hypot((ts[-1] - ts[0]) * 1000, end - start))
    return {
        "f0_start_khz": start,
        "f0_end_khz": end,
        "f0_min_khz": float(contour.min()),
        "f0_max_khz": float(contour.max()),
        "f0_mean_khz": float(contour.mean()),
        "f0_median_khz": float(np.median(contour)),
        "f0_sd_khz": float(contour.std()),
        "bandwidth_khz": float(contour.max() - contour.min()),
        "slope_khz_per_s": float(np.polyfit(ts, contour, 1)[0]),
        "sinuosity": arc / chord if chord > 0 else 1.0,
        "n_jumps": jumps,
        "harmonic_share": (
            float(np.nanmean(harmonic)) if np.isfinite(harmonic).any() else None
        ),  # share of frames with a harmonic, among frames where 2*f0 is inside the band
        "harmonic_observable_share": float(np.isfinite(harmonic).mean()),
        "tonality": float(np.mean(tonal)),
        "peak_freq_khz": float(contour[int(np.argmax(pw))]),
        "peak_snr_db": float(pw.max()),
        "mean_snr_db": float(pw.mean()),
        "n_contour_frames": len(f0),
    }


def recording_features(
    path: Path, calls: list[dict[str, Any]], cfg: DetectorConfig
) -> list[dict[str, Any]]:
    """Features for every call of one recording, using the detector's chunked noise floor."""
    info = sf.info(str(path))
    fs, total = info.samplerate, info.frames
    step, pad = int(cfg.chunk_s * fs), int(cfg.chunk_overlap_s * fs)
    out: list[dict[str, Any]] = []
    with sf.SoundFile(str(path), mode="r") as handle:
        for begin in range(0, total, step):
            owned = [c for c in calls if begin / fs <= c["start_s"] < (begin + step) / fs]
            if not owned:
                continue
            lo = max(0, begin - pad)
            handle.seek(lo)
            x = handle.read(min(total, begin + step + pad) - lo, dtype="float32", always_2d=True)
            f, t, db = whitened_spectrogram(x[:, 0], fs, cfg)
            t = t + lo / fs
            f_khz = (f / 1000).astype(np.float32)
            for c in owned:
                sel = (t >= c["start_s"]) & (t <= c["end_s"])
                feats = call_features(f_khz, t[sel], db[:, sel], cfg)
                row = {
                    "wav_path": c["wav_path"],
                    "call_index": c["call_index"],
                    "duration_ms": c["duration_ms"],
                    "contour_ok": feats is not None,
                }
                row.update(feats or {})
                out.append(row)
    return out


def _job(args: tuple[Path, list[dict[str, Any]], DetectorConfig]) -> list[dict[str, Any]]:
    return recording_features(*args)


def animal_day_table(
    calls: list[dict[str, Any]], recs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per recording: counts, rates, inter-call intervals and median features."""
    by_rec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        by_rec[c["wav_path"]].append(c)
    rows = []
    for r in recs:
        cs = sorted(by_rec[r["wav_path"]], key=lambda c: c["start_s"])
        conf = [c for c in cs if c.get("contour_ok") and c["peak_snr_db"] >= CONFIDENT_SNR_DB]
        minutes = r["duration_s"] / 60
        starts = np.array([c["start_s"] for c in conf])
        ici = np.diff(starts)
        row = {
            k: r[k]
            for k in ("wav_path", "animal_id", "marking", "sex", "litter_id", "dam_id",
                      "postnatal_age_days", "edit_suspected", "duration_s")
        }  # fmt: skip
        row.update(
            {
                "n_calls_all": len(cs),
                "n_calls": len(conf),
                "call_rate_per_min": len(conf) / minutes if minutes else None,
                "call_rate_all_per_min": len(cs) / minutes if minutes else None,
                "ici_median_s": float(np.median(ici)) if ici.size else None,
                "ici_short_share": float(np.mean(ici < 0.5)) if ici.size else None,
            }
        )
        # Harmonics only count where 2*f0 was inside the band for most of the call.
        observable = [c for c in conf if (c.get("harmonic_observable_share") or 0) >= 0.5]
        row["n_harmonic_observable_calls"] = len(observable)
        for k in FEATURES:
            pool = observable if k == "harmonic_share" else conf
            vals = [c[k] for c in pool if c.get(k) is not None]
            row[f"median_{k}"] = float(np.median(vals)) if vals else None
        rows.append(row)
    return rows


def run_features(
    cfg: SourceConfig, det: DetectorConfig, detect_run: Path, out: Path, workers: int
) -> dict[str, Any]:
    calls = pq.read_table(detect_run / "calls.parquet").to_pylist()
    recs = pq.read_table(detect_run / "recordings_detection.parquet").to_pylist()
    by_rec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        by_rec[c["wav_path"]].append(c)
    jobs = [(cfg.source_root / p, cs, det) for p, cs in by_rec.items()]
    feats: dict[tuple[str, int], dict[str, Any]] = {}
    with ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(_job, jobs):
            for row in rows:
                feats[row["wav_path"], row["call_index"]] = row
    merged = []
    for c in calls:
        f = feats.get((c["wav_path"], c["call_index"]), {"contour_ok": False})
        merged.append(
            {
                **{k: v for k, v in c.items() if k not in FEATURES},
                **{k: f.get(k) for k in FEATURES if k != "duration_ms"},
                "duration_ms": c["duration_ms"],
                "detector_peak_freq_khz": c["peak_freq_khz"],
                "detector_peak_snr_db": c["peak_snr_db"],
                "contour_ok": f["contour_ok"],
                "confident": bool(
                    f["contour_ok"] and (f.get("peak_snr_db") or 0) >= CONFIDENT_SNR_DB
                ),
            }
        )
    # Use contour SNR for the confidence flag; keep the detector's own SNR alongside.
    for m in merged:
        if m["peak_snr_db"] is None:
            m["peak_snr_db"] = m["detector_peak_snr_db"]
    day = animal_day_table(merged, recs)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(merged), out / "call_features.parquet")
    pq.write_table(pa.Table.from_pylist(day), out / "animal_day.parquet")
    summary = {
        "detect_run": str(detect_run),
        "calls": len(merged),
        "contour_ok": sum(m["contour_ok"] for m in merged),
        "confident": sum(m["confident"] for m in merged),
        "recordings": len(day),
        "confident_snr_db": CONFIDENT_SNR_DB,
        "jump_khz": JUMP_KHZ,
        "harmonic_share_mean": float(
            np.mean([m["harmonic_share"] for m in merged if m.get("harmonic_share") is not None])
        ),
        "f0_from_half_rule": f"sub-peak at f/2 >= {HALF_MIN_KHZ} kHz within {HALF_WITHIN_DB} dB",
    }
    write_json(out / "features_summary.json", summary)
    return summary
