"""M1: run the detector over all final pup recordings and write calls, background windows, QC."""

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rat_usv.config import SourceConfig
from rat_usv.data.avisoft import load_references
from rat_usv.data.linkage import Recording, final_pup_recordings, load_recordings
from rat_usv.provenance.io import write_json
from rat_usv.signal.detect import Call, DetectorConfig, detect_file
from rat_usv.signal.evaluate import match_calls

ARCHIVE = Path("Other Data/Data backup/Data.zip")
AGE_BANDS = ((1, 7), (8, 14), (15, 21))
META = ("wav_path", "animal_id", "marking", "sex", "litter_id", "dam_id", "postnatal_age_days")


def tuning_sample(recordings: list[Recording], references: set[str], seed: int = 1) -> list[str]:
    """The 36 recordings used to tune the detector (4 per litter x age band)."""
    rng = random.Random(seed)
    picked: list[str] = []
    for litter in ("E2L1", "E2L2", "E2L3"):
        for lo, hi in AGE_BANDS:
            pool = [
                r
                for r in recordings
                if r.litter_id == litter
                and r.postnatal_age_days is not None
                and lo <= r.postnatal_age_days <= hi
                and r.wav_path in references
            ]
            picked += [r.wav_path for r in rng.sample(pool, 4)]
    return picked


def background_windows(
    calls: list[Call], duration_s: float, n: int, length_s: float, guard_s: float, seed: int
) -> list[tuple[float, float]]:
    """Random windows at least guard_s away from any detected call (for negative controls)."""
    rng = np.random.default_rng(seed)
    busy = [(c.start_s - guard_s, c.end_s + guard_s) for c in calls]
    out: list[tuple[float, float]] = []
    for _ in range(n * 50):
        if len(out) == n:
            break
        start = float(rng.uniform(0, max(0.0, duration_s - length_s)))
        end = start + length_s
        if all(end <= a or start >= b for a, b in busy) and all(
            end <= a or start >= b for a, b in out
        ):
            out.append((round(start, 4), round(end, 4)))
    return sorted(out)


def _detect(args: tuple[Path, str, DetectorConfig]) -> tuple[str, list[Call], float]:
    root, wav_path, cfg = args
    calls, duration = detect_file(root / wav_path, cfg)
    return wav_path, calls, duration


def _rates(m: Any) -> dict[str, Any]:
    return {
        "tp": m.true_positive,
        "fp": m.false_positive,
        "fn": m.false_negative,
        "precision": m.precision,
        "recall": m.recall,
    }


def run_detection(
    cfg: SourceConfig, det: DetectorConfig, out: Path, workers: int, limit: int | None
) -> dict[str, Any]:
    if cfg.linkage_dir is None:
        raise ValueError("Config needs linkage_dir")
    everything = final_pup_recordings(load_recordings(cfg.linkage_dir))
    archive = cfg.source_root / ARCHIVE
    refs = load_references(archive, everything) if archive.is_file() else {}
    tuning = set(tuning_sample(everything, set(refs))) if refs else set()
    recs = everything[:limit] if limit else everything
    by_path = {r.wav_path: r for r in recs}

    call_rows: list[dict[str, Any]] = []
    bg_rows: list[dict[str, Any]] = []
    per_rec: list[dict[str, Any]] = []
    agree: dict[str, list[Any]] = defaultdict(list)
    jobs = [(cfg.source_root, r.wav_path, det) for r in recs]
    with ProcessPoolExecutor(workers) as pool:
        for i, (wav_path, calls, duration) in enumerate(pool.map(_detect, jobs)):
            rec = by_path[wav_path]
            meta = {k: getattr(rec, k) for k in META}
            meta["edit_suspected"] = rec.edit_suspected
            for n, c in enumerate(calls):
                call_rows.append({**meta, "call_index": n, **c.row()})
            for n, (a, b) in enumerate(background_windows(calls, duration, 10, 0.5, 0.2, i)):
                bg_rows.append({**meta, "window_index": n, "start_s": a, "end_s": b})
            row = {**meta, "duration_s": round(duration, 3), "n_calls": len(calls)}
            if wav_path in refs:
                m = match_calls(calls, refs[wav_path])
                row.update({"avisoft_calls": len(refs[wav_path]), **_rates(m)})
                split = "tuning" if wav_path in tuning else "held_out"
                age = rec.postnatal_age_days or 0
                band = next(f"P{lo}-P{hi}" for lo, hi in AGE_BANDS if lo <= age <= hi)
                agree[split].append(m)
                agree[f"{split}|{band}"].append(m)
            per_rec.append(row)

    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(call_rows), out / "calls.parquet")
    pq.write_table(pa.Table.from_pylist(bg_rows), out / "background_windows.parquet")
    pq.write_table(pa.Table.from_pylist(per_rec), out / "recordings_detection.parquet")

    def pooled(ms: list[Any]) -> dict[str, Any]:
        tp = sum(m.true_positive for m in ms)
        fp = sum(m.false_positive for m in ms)
        fn = sum(m.false_negative for m in ms)
        return {
            "recordings": len(ms),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "agreement_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "agreement_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }

    calls_by_age: dict[int, list[int]] = defaultdict(list)
    for row in per_rec:
        calls_by_age[int(row["postnatal_age_days"])].append(int(row["n_calls"]))
    summary = {
        "detector": det.model_dump(),
        "recordings": len(per_rec),
        "calls": len(call_rows),
        "background_windows": len(bg_rows),
        "recordings_with_avisoft_reference": len(refs),
        "avisoft_agreement": {k: pooled(v) for k, v in sorted(agree.items())},
        "avisoft_note": "2020 analyst exports are a reference, not ground truth; "
        "precision here counts calls the analyst skipped as errors.",
        "calls_per_recording_by_age": {
            f"P{age}": {
                "recordings": len(v),
                "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
                "zero_call_recordings": sum(x == 0 for x in v),
                "under_10_calls": sum(x < 10 for x in v),
            }
            for age, v in sorted(calls_by_age.items())
        },
        "tuning_recordings": sorted(tuning),
    }
    write_json(out / "detection_summary.json", summary)
    return summary
