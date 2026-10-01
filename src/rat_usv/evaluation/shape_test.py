"""H-M7a test exactly as pre-registered in decision 0018 (plus labelled post-hoc sensitivity)."""

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
from sklearn.model_selection import GroupKFold

from rat_usv.evaluation.controls import add_derived, feature_sets, regressor
from rat_usv.evaluation.m4 import fit_predict, paired_bootstrap, per_pup_mae, pooled
from rat_usv.evaluation.splits import load_splits
from rat_usv.representations import vae
from rat_usv.signal.features import SHAPE
from rat_usv.statistics.trajectory import fit as fit_trajectory

MIN_CALLS = 5


def shape_day_table(call_features: pd.DataFrame) -> pd.DataFrame:
    c = call_features[call_features["confident"] & call_features["u_index_khz"].notna()]
    g = c.groupby("wav_path")
    parts = [g[f].median().rename(f"shape_med_{f}") for f in SHAPE]
    parts += [g[f].quantile(0.9).rename(f"shape_p90_{f}") for f in SHAPE]
    out = pd.concat(parts, axis=1)
    out["n_shape_calls"] = g.size()
    return out[out["n_shape_calls"] >= MIN_CALLS]


def test_a(
    features_run: Path, controls_run: Path, images_run: Path, vae_runs: list[Path]
) -> dict[str, Any]:
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    splits = load_splits(controls_run / "splits.json", digest)
    m3 = json.loads((controls_run / "controls_summary.json").read_text())
    birth = {k: pd.Timestamp(v).date() for k, v in m3["birth_dates"].items()}
    day = add_derived(pd.read_parquet(controls_run / "animal_day_with_background.parquet"), birth)
    classical = feature_sets(day, [])["classical"]
    shape = shape_day_table(pd.read_parquet(features_run / "call_features.parquet"))
    shape_cols = [c for c in shape.columns if c.startswith("shape_")]
    day = day.merge(shape, left_on="wav_path", right_index=True, how="left")
    out: dict[str, Any] = {}

    # A1: direction with age (mixed models on per pup-day medians).
    a1 = {}
    for f in ("u_index_khz", "fm_residual_sd_khz", "fm_extrema_per_10ms"):
        col = f"shape_med_{f}"
        d = day.dropna(subset=[col]).assign(y=lambda x, c=col: x[c])
        r = fit_trajectory(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lin = smf.mixedlm("y ~ age + C(sex) + C(litter_id)", d, groups=d["animal_id"]).fit(
                reml=True
            )
        a1[f] = {"age_spline_lrt_p": r["age_lrt_p"], "linear_slope_per_day": round(float(lin.params["age"]), 4),
                 "linear_p": float(lin.pvalues["age"]), "n_animal_days": r["n_animal_days"]}  # fmt: skip
    rising = sum(v["linear_slope_per_day"] > 0 and v["linear_p"] < 0.05 for v in a1.values())
    out["A1_direction"] = {"features": a1, "rising_significant": rising, "supported": rising >= 2}

    # A2: classical + shape vs classical.
    preds = {k: np.full(len(day), np.nan) for k in ("classical", "classical_shape")}
    for fold in splits["leave_one_litter_out"]:
        test = (day["litter_id"] == fold["litter"]).to_numpy()
        preds["classical"][test] = fit_predict(day[~test], day[test], classical)
        preds["classical_shape"][test] = fit_predict(day[~test], day[test], classical + shape_cols)
    scored = day[["animal_id", "age"]].copy()
    for k, v in preds.items():
        scored[f"pred_{k}"] = v
    mae = {k: per_pup_mae(scored, f"pred_{k}") for k in preds}
    rng = np.random.default_rng(0)
    out["A2_shape_over_classical"] = paired_bootstrap(mae["classical"], mae["classical_shape"], rng)
    out["A2_per_pup_mae"] = {k: round(float(v.mean()), 3) for k, v in mae.items()}

    # A3: how much of the VAE gain do shape features absorb?
    call_idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    images = np.asarray(np.load(images_run / "calls_images.npy", mmap_mode="r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a3 = {}
    for run in vae_runs:
        run_seed = json.loads(next(run.glob("m4_*_summary.json")).read_text())["encoder"]["seed"]
        p = {k: np.full(len(day), np.nan) for k in ("classical_vae", "classical_shape_vae")}
        for fold in splits["leave_one_litter_out"]:
            litter = fold["litter"]
            model = vae.VAE().to(device)
            model.load_state_dict(torch.load(run / f"vae_holdout_{litter}.pt", map_location=device))
            emb = pooled(call_idx, vae.embed(model, images, device))
            d = day.merge(emb, left_on="wav_path", right_index=True, how="left")
            test = (d["litter_id"] == litter).to_numpy()
            p["classical_vae"][test] = fit_predict(d[~test], d[test], classical + list(emb.columns))
            p["classical_shape_vae"][test] = fit_predict(
                d[~test], d[test], classical + shape_cols + list(emb.columns)
            )
        sc = day[["animal_id", "age"]].copy()
        for k, v in p.items():
            sc[f"pred_{k}"] = v
        m2 = {k: per_pup_mae(sc, f"pred_{k}") for k in p}
        gain_over_classical = float((mae["classical"] - m2["classical_vae"]).mean())
        gain_over_shape = float((mae["classical_shape"] - m2["classical_shape_vae"]).mean())
        a3[f"vae_seed{run_seed}"] = {
            "vae_gain_over_classical": round(gain_over_classical, 3),
            "vae_gain_over_classical_shape": round(gain_over_shape, 3),
            "fraction_absorbed_by_shape": round(1 - gain_over_shape / gain_over_classical, 3),
            "vae_over_classical_shape": paired_bootstrap(
                mae["classical_shape"], m2["classical_shape_vae"], rng
            ),
        }
    out["A3_vae_gain_absorbed"] = a3
    out["A3_mean_fraction_absorbed"] = round(
        float(np.mean([v["fraction_absorbed_by_shape"] for v in a3.values()])), 3
    )
    out["status"] = "Test A is hypothesis-informed (not independent); see decision 0018"
    return out


FX_BASE = [
    "Call Length (s)",
    "Principal Frequency (kHz)",
    "Delta Freq (kHz)",
    "Slope (kHz/s)",
    "Peak Freq (kHz)",
]
FX_FM = ["Sinuosity", "Frequency Standard Deviation (kHz)"]
FX_LABELS = ["Trill", "Inverted_U", "Step_Up", "Step_Down", "Multi"]


def fragile_x_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    g = d.groupby("File")
    t = pd.DataFrame({"rat": g["Rat"].first(), "genotype": g["Genotype"].first(),
                      "age": g["PND"].first().str[1:].astype(float), "n": g.size()})  # fmt: skip
    for c in FX_BASE:
        t[f"base_{c}"] = g[c].median()
    t["base_log_n"] = np.log(t["n"])
    for c in FX_FM:
        t[f"fm_med_{c}"] = g[c].median()
        t[f"fm_p90_{c}"] = g[c].quantile(0.9)
    for lab in FX_LABELS:
        t[f"fm_share_{lab}"] = g["Label"].apply(lambda s, lab=lab: float((s == lab).mean()))
    return t[t["n"] >= MIN_CALLS]


def _cv_mae(t: pd.DataFrame, cols: list[str], folds: int = 5) -> pd.Series:
    pred = np.full(len(t), np.nan)
    X, y, g = t[cols].to_numpy(float), t["age"].to_numpy(float), t["rat"].to_numpy()
    for train, test in GroupKFold(min(folds, len(set(g)))).split(X, y, g):
        model = regressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train], y[train], groups=g[train])
        pred[test] = model.predict(X[test])
    return pd.Series(np.abs(pred - y)).groupby(g).mean()


def test_b(path: Path) -> dict[str, Any]:
    t = fragile_x_table(path)
    base = [c for c in t.columns if c.startswith("base_")]
    fm = [c for c in t.columns if c.startswith("fm_")]
    longitudinal = t.groupby("rat")["age"].transform(lambda s: (s != 10).all())
    out: dict[str, Any] = {"pup_days": int(len(t)), "rats": int(t["rat"].nunique())}
    rng = np.random.default_rng(0)
    for label, sub in (("B1_WT", t[t["genotype"] == "WT"]), ("B2_KO", t[t["genotype"] == "KO"]),
                       ("posthoc_WT_longitudinal_only", t[(t["genotype"] == "WT") & longitudinal]),
                       ("posthoc_KO_longitudinal_only", t[(t["genotype"] == "KO") & longitudinal])):  # fmt: skip
        mb, mf = _cv_mae(sub, base), _cv_mae(sub, base + fm)
        out[label] = {"rats": int(sub["rat"].nunique()), "pup_days": int(len(sub)),
                      "mae_base": round(float(mb.mean()), 3), "mae_base_fm": round(float(mf.mean()), 3),
                      "fm_gain": paired_bootstrap(mb, mf, rng)}  # fmt: skip
    wt = t[t["genotype"] == "WT"]
    out["B3_WT_direction"] = {
        "sinuosity_median_vs_age_rho": round(float(stats.spearmanr(wt["fm_med_Sinuosity"], wt["age"]).statistic), 3),
        "modulated_share_vs_age_rho": round(
            float(stats.spearmanr(wt[[f"fm_share_{x}" for x in FX_LABELS]].sum(axis=1), wt["age"]).statistic), 3),
    }  # fmt: skip
    out["B1_supported"] = out["B1_WT"]["fm_gain"]["ci95"][0] > 0
    out["note"] = ("P10 pups in this dataset are a separate cohort recorded only at P10, so age is confounded "
                   "with cohort; post-hoc rows exclude them.")  # fmt: skip
    return out
