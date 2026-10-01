"""H2 / H2b (protocol v1): trajectory structure of held-out pups and acoustic age vs weight.

All geometry is computed for pups of the litter each fold's encoder did NOT see, in that fold's
space. Classical spaces are standardised with training-litter statistics only.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
import torch
from scipy import stats
from scipy.spatial.distance import cdist, pdist

from rat_usv.evaluation.controls import add_derived, feature_sets
from rat_usv.evaluation.splits import load_splits
from rat_usv.representations import encoders, vae

MIN_CALLS = 5


def centroids(index: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(z, columns=[f"z{i:02d}" for i in range(z.shape[1])])
    df["wav_path"] = index["wav_path"].to_numpy()
    counts = df.groupby("wav_path").size()
    mean = df.groupby("wav_path").mean()
    return mean[counts >= MIN_CALLS]


def distance_age_rho(X: np.ndarray, ages: np.ndarray) -> float:
    """Spearman correlation between pairwise distance and |age difference| within one pup."""
    if len(ages) < 4:
        return float("nan")
    return float(stats.spearmanr(pdist(X), pdist(ages[:, None])).statistic)


def alignment_error(Xa: np.ndarray, aa: np.ndarray, Xb: np.ndarray, ab: np.ndarray) -> float:
    """Mean |age of pup B's nearest day - age of pup A's day| over A's days."""
    nearest = cdist(Xa, Xb).argmin(axis=1)
    return float(np.abs(ab[nearest] - aa).mean())


def space_metrics(
    pups: dict[str, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator, n_null: int = 200
) -> dict[str, Any]:
    """pups: animal_id -> (X [days x dims], ages). Pups share one space (same held-out litter)."""
    rhos, null_rhos, align, null_align = [], [], [], []
    names = sorted(pups)
    for a in names:
        X, ages = pups[a]
        rhos.append(distance_age_rho(X, ages))
        null_rhos.append(
            np.nanmean([distance_age_rho(X, rng.permutation(ages)) for _ in range(20)])
        )
        for b in names:
            if a == b:
                continue
            Xb, ab = pups[b]
            align.append(alignment_error(X, ages, Xb, ab))
            null_align.append(
                np.mean([alignment_error(X, ages, Xb, rng.permutation(ab)) for _ in range(20)])
            )
    out = {
        "n_pups": len(names),
        "distance_age_rho_median": float(np.nanmedian(rhos)),
        "distance_age_rho_null": float(np.nanmedian(null_rhos)),
        "pups_rho_above_null": int(np.sum(np.array(rhos) > np.array(null_rhos))),
        "alignment_error_days": float(np.mean(align)) if align else None,
        "alignment_error_null": float(np.mean(null_align)) if null_align else None,
        "pairs": len(align),
    }
    stacked = np.vstack([X for X, _ in pups.values()])
    if stacked.shape[1] > 1 and stacked.shape[0] > stacked.shape[1]:
        ev = np.linalg.svd(stacked - stacked.mean(0), compute_uv=False) ** 2
        ev = ev / ev.sum()
        out["pc1_variance_share"] = float(ev[0])
        out["pcs_for_90pct"] = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
    return out


def _standardise(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    med = train[cols].median()
    mu = train[cols].fillna(med).mean()
    sd = train[cols].fillna(med).std().replace(0, 1)
    return ((test[cols].fillna(med) - mu) / sd).to_numpy(float)


def run_h2(
    m4_runs: list[Path], images_run: Path, controls_run: Path, seed: int = 0
) -> dict[str, Any]:
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    splits = load_splits(controls_run / "splits.json", digest)
    m3 = json.loads((controls_run / "controls_summary.json").read_text())
    birth = {k: pd.Timestamp(v).date() for k, v in m3["birth_dates"].items()}
    day = add_derived(pd.read_parquet(controls_run / "animal_day_with_background.parquet"), birth)
    day = day[day["n_calls"] >= MIN_CALLS]
    classical = feature_sets(day, [])["classical"]
    call_idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    images = np.load(images_run / "calls_images.npy", mmap_mode="r")
    bg_idx = pq.read_table(images_run / "background_index.parquet").to_pandas()
    bg_idx = bg_idx.merge(
        day[["wav_path", "litter_id"]].drop_duplicates(),
        on="wav_path",
        how="left",
        suffixes=("_x", ""),
    )
    bg_images = np.load(images_run / "background_images.npy", mmap_mode="r")
    bg_bands = [c for c in day.columns if c.startswith("bg_") and c.endswith("khz")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    results: dict[str, list[dict[str, Any]]] = {}

    def collect(space: str, litter: str, rows: pd.DataFrame, X: np.ndarray) -> None:
        pups = {}
        for animal, idx in rows.groupby("animal_id").indices.items():
            pups[animal] = (X[idx], rows["age"].to_numpy(float)[idx])
        results.setdefault(space, []).append({"litter": litter, **space_metrics(pups, rng)})

    for fold in splits["leave_one_litter_out"]:
        litter = fold["litter"]
        test_rows = day[day["litter_id"] == litter].reset_index(drop=True)
        train_rows = day[day["litter_id"] != litter]
        collect(
            "classical_multivariate",
            litter,
            test_rows,
            _standardise(train_rows, test_rows, classical),
        )
        for single in ("median_f0_median_khz", "median_duration_ms", "log_call_rate"):
            collect(
                f"classical_{single}",
                litter,
                test_rows,
                _standardise(train_rows, test_rows, [single]),
            )
        # Session control: the same geometry on call-free background (band powers, and embeddings).
        collect(
            "background_bands", litter, test_rows, _standardise(train_rows, test_rows, bg_bands)
        )
        bg_held = (bg_idx["litter_id"] == litter).to_numpy()
        bg_sub_idx = bg_idx[bg_held].reset_index(drop=True)
        bg_sub_img = np.asarray(bg_images[np.flatnonzero(bg_held)])
        held = (call_idx["litter_id"] == litter).to_numpy()
        sub_idx = call_idx[held].reset_index(drop=True)
        sub_img = np.asarray(images[np.flatnonzero(held)])
        for run in m4_runs:
            summary = json.loads(next(run.glob("m4_*_summary.json")).read_text())
            kind, run_seed = summary["rung"], summary["encoder"]["seed"]
            state = torch.load(run / f"{kind}_holdout_{litter}.pt", map_location=device)
            if kind == "vae":
                model = vae.VAE().to(device)
                model.load_state_dict(state)
                z = vae.embed(model, sub_img, device)
            else:
                enc = encoders.Encoder().to(device)
                enc.load_state_dict(state)
                z = encoders.embed_encoder(enc, sub_img, device)
            cent = centroids(sub_idx, z)
            rows = test_rows[test_rows["wav_path"].isin(cent.index)].reset_index(drop=True)
            X = cent.loc[rows["wav_path"]].to_numpy()
            X = (X - X.mean(0)) / X.std(0).clip(1e-6)  # scale-free per space; no age used
            collect(f"{kind}_seed{run_seed}", litter, rows, X)
            zb = (vae.embed(model, bg_sub_img, device) if kind == "vae"
                  else encoders.embed_encoder(enc, bg_sub_img, device))  # fmt: skip
            bcent = centroids(bg_sub_idx, zb)
            brows = test_rows[test_rows["wav_path"].isin(bcent.index)].reset_index(drop=True)
            B = bcent.loc[brows["wav_path"]].to_numpy()
            B = (B - B.mean(0)) / B.std(0).clip(1e-6)
            collect(f"background_{kind}_seed{run_seed}", litter, brows, B)

    def pooled(entries: list[dict[str, Any]]) -> dict[str, Any]:
        w = np.array([e["n_pups"] for e in entries], float)
        keys = ["distance_age_rho_median", "distance_age_rho_null", "alignment_error_days",
                "alignment_error_null", "pc1_variance_share", "pcs_for_90pct"]  # fmt: skip
        out = {
            k: round(float(np.average([e.get(k) or np.nan for e in entries], weights=w)), 3)
            for k in keys
        }
        out["pups_rho_above_null"] = int(sum(e["pups_rho_above_null"] for e in entries))
        out["n_pups"] = int(w.sum())
        return out

    return {"by_space": {k: pooled(v) for k, v in results.items()}, "by_litter": results,
            "min_calls_per_animal_day": MIN_CALLS}  # fmt: skip


def run_h2b(m4_runs: list[Path], controls_run: Path, linkage_dir: Path) -> dict[str, Any]:
    """Does predicted ('acoustic') age run ahead in heavier pups, at the same age and litter?

    weight_z = weight standardised within litter x postnatal day (so it is uncorrelated with age).
    Model: (predicted - true age) ~ weight_z + cr(age) + sex, random intercept per pup.
    """
    rec = pd.read_csv(linkage_dir / "recordings.csv")
    rec = rec[rec["in_final_record_list"] & rec["animal_id"].notna()][["wav_path", "weight_g"]]
    out: dict[str, Any] = {}
    for run in m4_runs:
        summary = json.loads(next(run.glob("m4_*_summary.json")).read_text())
        p = pd.read_parquet(run / "predictions_lolo.parquet").merge(rec, on="wav_path")
        day = pd.read_parquet(controls_run / "animal_day_with_background.parquet")[
            ["wav_path", "sex"]
        ]
        p = p.merge(day, on="wav_path")
        p["weight_z"] = p.groupby(["litter_id", "age"])["weight_g"].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1)
        )
        res = {}
        for target in ("classical", "learned", "combined"):
            p["err"] = p[f"pred_{target}"] - p["age"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm(
                    'err ~ weight_z + cr(age, df=4, constraints="center") + C(sex)',
                    p,
                    groups=p["animal_id"],
                ).fit(reml=True)
            beta, ci = m.params["weight_z"], m.conf_int().loc["weight_z"].tolist()
            res[target] = {"days_per_weight_sd": round(float(beta), 3),
                           "ci95": [round(float(ci[0]), 3), round(float(ci[1]), 3)],
                           "p": float(m.pvalues["weight_z"])}  # fmt: skip
        out[f"{summary['rung']}_seed{summary['encoder']['seed']}"] = res
    return out


def json_safe(obj: Any) -> Any:
    """Replace NaN/inf with None recursively (missing metrics stay explicit, never zero)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
