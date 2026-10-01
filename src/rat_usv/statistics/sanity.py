"""Sanity check: our per-recording summaries against the 2020 Avisoft analysis."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from rat_usv.config import SourceConfig
from rat_usv.data.avisoft import load_references
from rat_usv.data.linkage import final_pup_recordings, load_recordings
from rat_usv.signal.pipeline import ARCHIVE


def avisoft_sanity(cfg: SourceConfig, features_run: Path) -> dict[str, Any]:
    """Spearman agreement of call counts and median frequency per recording."""
    assert cfg.linkage_dir is not None
    recs = final_pup_recordings(load_recordings(cfg.linkage_dir))
    refs = load_references(cfg.source_root / ARCHIVE, recs)
    day = pd.read_parquet(features_run / "animal_day.parquet").set_index("wav_path")
    rows = []
    for path, calls in refs.items():
        if path not in day.index:
            continue
        freqs = [c.peak_freq_khz for c in calls if c.peak_freq_khz is not None]
        rows.append(
            {
                "avisoft_n": len(calls),
                "ours_n": day.at[path, "n_calls"],
                "ours_n_all": day.at[path, "n_calls_all"],
                "avisoft_freq": float(np.median(freqs)) if freqs else None,
                "ours_freq": day.at[path, "median_f0_median_khz"],
            }
        )
    d = pd.DataFrame(rows)
    out: dict[str, Any] = {"recordings": len(d)}
    for a, b in (
        ("avisoft_n", "ours_n"),
        ("avisoft_n", "ours_n_all"),
        ("avisoft_freq", "ours_freq"),
    ):
        sub = d[[a, b]].dropna()
        rho, p = stats.spearmanr(sub[a], sub[b])
        out[f"{a}~{b}"] = {"n": len(sub), "spearman_rho": round(float(rho), 3), "p": float(p)}
    sub = d[["avisoft_freq", "ours_freq"]].dropna()
    out["median_freq_difference_khz"] = round(
        float((sub["ours_freq"] - sub["avisoft_freq"]).median()), 2
    )
    return out
