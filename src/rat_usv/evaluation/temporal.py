"""M5 / H3: does call timing and order add age information beyond single-call structure?

Owner decision: hand-built features first (no sequence model). Per pup-day, from confident
calls in time order:
- order-free timing: inter-call interval (ICI, start-to-start) quantiles and CV, silent-gap
  median, bouts (a gap > BOUT_GAP_S starts a new bout; fixed a priori);
- order-dependent sequence: 8x8 transition matrix between the fold's VAE call families
  (M7, k-means fit on training litters), conditional entropy H(next | current), same-family
  repeat rate, and the local variation (LV) of successive intervals.
Control: sequence features recomputed after shuffling call order within each recording.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from rat_usv.evaluation.controls import add_derived, feature_sets
from rat_usv.evaluation.explain import K, fold_families
from rat_usv.evaluation.m4 import fit_predict, paired_bootstrap, per_pup_mae, pooled
from rat_usv.evaluation.splits import load_splits

BOUT_GAP_S = 0.5
N_SHUFFLES = 5
TIMING = [
    "ici_q10",
    "ici_q50",
    "ici_q90",
    "ici_cv",
    "gap_q50",
    "bouts_per_min",
    "calls_per_bout",
    "bout_dur_q50",
]
SEQUENCE = [f"t_{a}{b}" for a in range(K) for b in range(K)] + [
    "seq_entropy",
    "repeat_rate",
    "ici_lv",
]


def timing_features(starts: np.ndarray, ends: np.ndarray, minutes: float) -> dict[str, float]:
    out = dict.fromkeys(TIMING, np.nan)
    if len(starts) < 3:
        return out
    ici = np.diff(starts)
    gaps = starts[1:] - ends[:-1]
    new_bout = np.concatenate([[True], gaps > BOUT_GAP_S])
    bout_id = np.cumsum(new_bout)
    sizes = np.bincount(bout_id)[1:]
    bstart = pd.Series(starts).groupby(bout_id).min()
    bend = pd.Series(ends).groupby(bout_id).max()
    out.update({
        "ici_q10": float(np.quantile(ici, 0.1)), "ici_q50": float(np.median(ici)),
        "ici_q90": float(np.quantile(ici, 0.9)), "ici_cv": float(ici.std() / ici.mean()),
        "gap_q50": float(np.median(gaps)), "bouts_per_min": float(len(sizes) / minutes),
        "calls_per_bout": float(sizes.mean()), "bout_dur_q50": float(np.median(bend - bstart)),
    })  # fmt: skip
    return out


def sequence_features(families: np.ndarray, starts: np.ndarray) -> dict[str, float]:
    out = dict.fromkeys(SEQUENCE, np.nan)
    if len(families) < 3:
        return out
    T = np.zeros((K, K))
    for a, b in zip(families[:-1], families[1:], strict=True):
        T[a, b] += 1
    total = T.sum()
    for a in range(K):
        for b in range(K):
            out[f"t_{a}{b}"] = float(T[a, b] / total)
    row = T.sum(axis=1, keepdims=True)
    P = np.divide(T, row, out=np.zeros_like(T), where=row > 0)
    w = row.ravel() / total
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(P > 0, P * np.log2(P), 0.0), axis=1)
    out["seq_entropy"] = float((w * h).sum())
    out["repeat_rate"] = float(np.trace(T) / total)
    ici = np.diff(starts)
    if len(ici) >= 2:
        a, b = ici[:-1], ici[1:]
        out["ici_lv"] = float(np.mean(3 * (a - b) ** 2 / (a + b) ** 2))
    return out


def recording_table(calls: pd.DataFrame, families: np.ndarray, minutes: pd.Series,
                    rng: np.random.Generator | None) -> pd.DataFrame:  # fmt: skip
    """Per recording features; if rng is given, call order (families and intervals) is shuffled."""
    rows = []
    c = calls.assign(family=families).sort_values(["wav_path", "start_s"])
    for path, g in c.groupby("wav_path", sort=False):
        starts, ends, fam = g["start_s"].to_numpy(), g["end_s"].to_numpy(), g["family"].to_numpy()
        row = {"wav_path": path, **timing_features(starts, ends, float(minutes.get(path, 5.0)))}
        if rng is not None:
            perm = rng.permutation(len(fam))
            fam = fam[perm]
            ici = np.diff(starts)
            starts = np.concatenate([[starts[0]], starts[0] + np.cumsum(rng.permutation(ici))])
        rows.append(row | sequence_features(fam, starts))
    return pd.DataFrame(rows).set_index("wav_path")


COMPACT = ["seq_entropy", "repeat_rate", "ici_lv"]  # post-hoc sensitivity set (decision 0017)


def run_temporal(
    vae_runs: list[Path], images_run: Path, controls_run: Path, seed: int = 0, compact: bool = False
) -> dict[str, Any]:
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    splits = load_splits(controls_run / "splits.json", digest)
    m3 = json.loads((controls_run / "controls_summary.json").read_text())
    birth = {k: pd.Timestamp(v).date() for k, v in m3["birth_dates"].items()}
    day = add_derived(pd.read_parquet(controls_run / "animal_day_with_background.parquet"), birth)
    classical = feature_sets(day, [])["classical"]
    minutes = day.set_index("wav_path")["duration_s"] / 60
    calls = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    images = np.asarray(np.load(images_run / "calls_images.npy", mmap_mode="r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_cols = COMPACT if compact else SEQUENCE
    out: dict[str, Any] = {"bout_gap_s": BOUT_GAP_S, "n_shuffles": N_SHUFFLES, "runs": {},
                           "sequence_set": "compact (post hoc)" if compact else "full (pre-specified)"}  # fmt: skip
    for run in vae_runs:
        run_seed = json.loads(next(run.glob("m4_*_summary.json")).read_text())["encoder"]["seed"]
        rng = np.random.default_rng(seed + run_seed)
        names = ["base", "timing", "sequence"] + [f"shuffled_{i}" for i in range(N_SHUFFLES)]
        preds = {k: np.full(len(day), np.nan) for k in names}
        for fold in splits["leave_one_litter_out"]:
            litter = fold["litter"]
            z, fam = fold_families(run, images, calls, litter, device, run_seed)
            emb = pooled(calls, z)
            real = recording_table(calls, fam, minutes, None)
            d = day.merge(emb, left_on="wav_path", right_index=True, how="left").merge(
                real, left_on="wav_path", right_index=True, how="left")  # fmt: skip
            test = (d["litter_id"] == litter).to_numpy()
            base = classical + list(emb.columns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds["base"][test] = fit_predict(d[~test], d[test], base)
                preds["timing"][test] = fit_predict(d[~test], d[test], base + TIMING)
                preds["sequence"][test] = fit_predict(d[~test], d[test], base + TIMING + seq_cols)
                for i in range(N_SHUFFLES):
                    shuf = recording_table(calls, fam, minutes, rng)[seq_cols]
                    ds = d.drop(columns=seq_cols).merge(
                        shuf, left_on="wav_path", right_index=True, how="left"
                    )
                    preds[f"shuffled_{i}"][test] = fit_predict(
                        ds[~test], ds[test], base + TIMING + seq_cols
                    )
            del z
        scored = day[["animal_id", "age"]].copy()
        for k, v in preds.items():
            scored[f"pred_{k}"] = v
        mae = {k: per_pup_mae(scored, f"pred_{k}") for k in names}
        shuffled = pd.concat([mae[f"shuffled_{i}"] for i in range(N_SHUFFLES)], axis=1).mean(axis=1)
        brng = np.random.default_rng(0)
        out["runs"][f"vae_seed{run_seed}"] = {
            "per_pup_mae_days": {k: round(float(mae[k].mean()), 3) for k in ("base", "timing", "sequence")}
            | {"sequence_shuffled_mean": round(float(shuffled.mean()), 3)},
            "timing_gain_over_base": paired_bootstrap(mae["base"], mae["timing"], brng),
            "sequence_gain_over_timing": paired_bootstrap(mae["timing"], mae["sequence"], brng),
            "real_order_vs_shuffled_order": paired_bootstrap(shuffled, mae["sequence"], brng),
        }  # fmt: skip
    return out
