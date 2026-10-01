"""M9: data-driven discovery of developmental call structure (decisions 0020, 0021).

Rotating discovery/confirmation by litter. For each held-out litter L:
1. discovery on the other two litters only: every fitted step (fPCA bases, call families,
   feature selection, inner decoding) sees discovery rows only, and `_discovery_rows` asserts
   that L is absent;
2. the predictions are written to `m9_predictions_holdout_<L>.json` and its SHA-256 is recorded
   before any confirmation code reads L;
3. confirmation on L tests the locked predictions as they are.
"""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from rat_usv.evaluation.controls import add_derived, feature_sets
from rat_usv.evaluation.m4 import fit_predict, paired_bootstrap, per_pup_mae
from rat_usv.evaluation.shape_test import shape_day_table
from rat_usv.evaluation.splits import load_splits
from rat_usv.provenance.io import sha256_file, write_json
from rat_usv.signal.curves import N_POINTS

LITTERS = ("E2L1", "E2L2", "E2L3")
FDR_Q = 0.05
MAX_MODES = 8
VAR_EXPLAINED = 0.90
MIN_CALLS = 5  # pup-day medians (protocol v1)
MIN_CALLS_VAR = 10  # pup-day dispersion (decision 0020)
N_FAMILIES = 8
FAMILY_INPUT = ("log_duration", "f0_median_khz", "bandwidth_khz")
WEAK_ROTATION = "E2L2"  # 4 pups


# ---------------------------------------------------------------- data


def load_calls(features_run: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Confident calls with stored curves; `row` indexes the curve arrays."""
    feats = pd.read_parquet(features_run / "call_features.parquet")
    index = pd.read_parquet(features_run / "call_curves_index.parquet")
    calls = feats[feats["confident"]].merge(index, on=["wav_path", "call_index"], how="inner")
    with np.load(features_run / "call_curves.npz") as z:
        curves = {k: z[k] for k in z.files}
    calls["log_duration"] = np.log(calls["duration_ms"])
    calls["age"] = calls["postnatal_age_days"].astype(float)
    return calls.reset_index(drop=True), curves


def load_days(controls_run: Path, features_run: Path) -> tuple[pd.DataFrame, list[str]]:
    """Pup-day table with the M9 baseline: classical + six H-M7a shape features (median, p90)."""
    m3 = json.loads((controls_run / "controls_summary.json").read_text())
    birth = {k: pd.Timestamp(v).date() for k, v in m3["birth_dates"].items()}
    day = add_derived(pd.read_parquet(controls_run / "animal_day_with_background.parquet"), birth)
    shape = shape_day_table(pd.read_parquet(features_run / "call_features.parquet"))
    day = day.merge(shape, left_on="wav_path", right_index=True, how="left")
    baseline = feature_sets(day, [])["classical"] + [
        c for c in shape.columns if c.startswith("shape_")
    ]
    return day, baseline


# ---------------------------------------------------------------- WP2 per-call features


def beyond_contour(calls: pd.DataFrame, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    """Envelope timing and modulation, harmonic ratio, tonality profile, nonlinear phenomena.

    No fitted statistics: every value is a fixed function of one call's curves.
    """
    env = curves["envelope"][calls["row"].to_numpy()].astype(float)
    peak = env.max(1, keepdims=True)
    near = env >= peak - 3.0  # within 3 dB of the call's peak level
    first = near.argmax(1)
    last = N_POINTS - 1 - near[:, ::-1].argmax(1)
    mid = env[:, 3:29]  # 10-90 % of the call
    tau = np.linspace(-1, 1, mid.shape[1])
    coef = np.polyfit(tau, mid.T, 1)
    resid = mid - (coef[0][:, None] * tau + coef[1][:, None])
    harm = curves["harmonic_ratio"][calls["row"].to_numpy()].astype(float)
    tonal = curves["tonality"][calls["row"].to_numpy()].astype(float)
    obs = (calls["harmonic_observable_share"].fillna(0) >= 0.5).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        harm_med = np.where(obs, np.nanmedian(harm, 1), np.nan)
    return pd.DataFrame(
        {
            "env_rise_frac": first / (N_POINTS - 1),
            "env_fall_frac": (N_POINTS - 1 - last) / (N_POINTS - 1),
            "env_am_depth_db": np.percentile(resid, 90, axis=1) - np.percentile(resid, 10, axis=1),
            "harmonic_ratio_db": harm_med,
            "tonality_start": tonal[:, :11].mean(1),
            "tonality_middle": tonal[:, 11:21].mean(1),
            "tonality_end": tonal[:, 21:].mean(1),
            "has_subharmonic": (calls["n_subharmonic_events"] > 0).astype(float).to_numpy(),
            "has_biphonation": (calls["n_biphonation_events"] > 0).astype(float).to_numpy(),
            "has_jump": (calls["n_jumps"] > 0).astype(float).to_numpy(),
        },
        index=calls.index,
    )


# ---------------------------------------------------------------- WP1 fPCA


def legendre_templates(k: int) -> np.ndarray:
    """Orthonormal discrete Legendre polynomials P1..Pk on N_POINTS (rows): naming templates."""
    tau = np.linspace(-1, 1, N_POINTS)
    t = np.stack([np.polynomial.legendre.Legendre.basis(i)(tau) for i in range(1, k + 1)])
    q, _ = np.linalg.qr(np.vstack([np.ones(N_POINTS), t]).T)
    return np.asarray(q[:, 1:].T)


def fpca(x: np.ndarray) -> dict[str, Any]:
    """PCA of per-call curves with each call's own mean removed (shape only).

    Modes are named by matching to Legendre templates (data-free, Hungarian on |cos|) and
    sign-flipped so the template cosine is positive, so names are comparable across rotations.
    """
    xc = x - x.mean(1, keepdims=True)
    mean = xc.mean(0)
    _, s, vt = np.linalg.svd(xc - mean, full_matrices=False)
    var = s**2 / (s**2).sum()
    k = min(MAX_MODES, int(np.searchsorted(np.cumsum(var), VAR_EXPLAINED)) + 1)
    modes = vt[:k]
    tmpl = legendre_templates(MAX_MODES)
    cos = modes @ tmpl.T
    rows, cols = optimize.linear_sum_assignment(-np.abs(cos))
    names, signed = [], []
    for r, c in zip(rows, cols, strict=True):
        sign = 1.0 if cos[r, c] >= 0 else -1.0
        signed.append(sign * modes[r])
        names.append(f"P{c + 1}")
    return {"mean": mean, "modes": np.array(signed), "names": names,
            "var": var[:k],
        "sd": s[:k] / np.sqrt(len(x)), "template_cos": [float(abs(cos[r, c])) for r, c in zip(rows, cols, strict=True)]}  # fmt: skip


def project(x: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    xc = x - x.mean(1, keepdims=True)
    return np.asarray((xc - basis["mean"]) @ basis["modes"].T)


# ---------------------------------------------------------------- pup-day aggregation


def mad_sd(x: pd.Series) -> float:
    v = x.dropna().to_numpy()
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def day_candidates(
    calls: pd.DataFrame, per_call: pd.DataFrame, families: np.ndarray
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Candidate pup-day features by block. `per_call` holds fPCA scores and WP2 features."""
    c = pd.concat([calls[["wav_path", "f0_median_khz", "log_duration"]], per_call], axis=1)
    c["family"] = families
    g = c.groupby("wav_path")
    n = g.size()
    blocks: dict[str, list[str]] = {
        "f0_modes": [],
        "env_modes": [],
        "beyond": [],
        "variability": [],
    }
    parts = []
    for col in per_call.columns:
        if col.startswith("f0m_") or col.startswith("envm_"):
            parts.append(g[col].median().rename(f"{col}_med"))
            blocks["f0_modes" if col.startswith("f0m_") else "env_modes"].append(f"{col}_med")
        elif col.startswith(("has_",)):
            parts.append(g[col].mean().rename(f"share_{col[4:]}"))
            blocks["beyond"].append(f"share_{col[4:]}")
        else:
            parts.append(g[col].median().rename(f"{col}_med"))
            blocks["beyond"].append(f"{col}_med")
    out = pd.concat(parts, axis=1)
    out.loc[n < MIN_CALLS] = np.nan
    f0_modes = [x for x in per_call.columns if x.startswith("f0m_")][:3]
    var_cols = ["f0_median_khz", "log_duration", *f0_modes]
    var = pd.concat([g[v].agg(mad_sd).rename(f"var_{v}") for v in var_cols], axis=1)
    # Within-family dispersion: call-weighted mean of per-family MAD-SD (families with >= 3 calls).
    for v in ("f0_median_khz", f0_modes[0]):
        fam = c.groupby(["wav_path", "family"])[v].agg(["size", mad_sd]).reset_index()
        fam = fam[fam["size"] >= 3]
        w = fam.groupby("wav_path").apply(
            lambda d: float(np.average(d["mad_sd"], weights=d["size"])), include_groups=False
        )
        var[f"var_within_family_{v}"] = w
    var.loc[n < MIN_CALLS_VAR] = np.nan
    blocks["variability"] = list(var.columns)
    return out.join(var, how="outer"), blocks


# ---------------------------------------------------------------- statistics


def age_trend(d: pd.DataFrame, col: str, litter_term: bool) -> dict[str, Any]:
    """Random-intercept (pup) model of the standardised feature on age; slope in SD/day.

    Fallbacks (decision 0021): a constant feature gives p = NaN (never significant); if the
    mixed model is singular, OLS with pup-clustered standard errors is used instead.
    """
    sub = d.dropna(subset=[col])
    sd = float(sub[col].std()) if len(sub) > 1 else 0.0
    base = {"n_pup_days": int(len(sub)), "n_pups": int(sub["animal_id"].nunique())}
    if not sd > 0:
        return base | {"slope_sd_per_day": 0.0, "p": float("nan"), "model": "constant"}
    sub = sub.assign(y=(sub[col] - sub[col].mean()) / sd)
    rhs = "age + C(sex)" + (" + C(litter_id)" if litter_term else "")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(f"y ~ {rhs}", sub, groups=sub["animal_id"]).fit(reml=True)
            kind = "mixed"
        except (np.linalg.LinAlgError, ValueError):
            m = smf.ols(f"y ~ {rhs}", sub).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["animal_id"])[0]}
            )
            kind = "ols_cluster_pup"
    return base | {"slope_sd_per_day": float(m.params["age"]), "p": float(m.pvalues["age"]),
                   "model": kind}  # fmt: skip


def bh(pvals: list[float]) -> list[bool]:
    """Benjamini-Hochberg over the finite p-values; NaN is never significant."""
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(p.size, dtype=bool)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], alpha=FDR_Q, method="fdr_bh")[0]
    return [bool(x) for x in out]


def inner_gain(d: pd.DataFrame, base: list[str], extra: list[str]) -> float:
    """Discovery-only decoding check: train on one discovery litter, test on the other."""
    pa, pb = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for litter in d["litter_id"].unique():
        test = (d["litter_id"] == litter).to_numpy()
        pa[test] = fit_predict(d[~test], d[test], base)
        pb[test] = fit_predict(d[~test], d[test], base + extra)
    s = d[["animal_id", "age"]].assign(pa=pa, pb=pb)
    return float((per_pup_mae(s, "pa") - per_pup_mae(s, "pb")).mean())


def _discovery_rows(df: pd.DataFrame, held_out: str) -> pd.DataFrame:
    out = df[df["litter_id"] != held_out]
    assert held_out not in set(out["litter_id"]), "held-out litter leaked into discovery"
    return out


# ---------------------------------------------------------------- rotation


def per_call_table(
    calls: pd.DataFrame, curves: dict[str, np.ndarray], wp2: pd.DataFrame, held_out: str
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """fPCA bases and call families fitted on discovery calls; applied to all calls."""
    disc = _discovery_rows(calls, held_out).index.to_numpy()
    rows = calls["row"].to_numpy()
    f0 = curves["f0"][rows].astype(float)
    env = curves["envelope"][rows].astype(float)
    bases = {"f0": fpca(f0[disc]), "env": fpca(env[disc])}
    per_call = wp2.copy()
    for name, x, prefix in (("f0", f0, "f0m"), ("env", env, "envm")):
        scores = project(x, bases[name])
        for j, mode in enumerate(bases[name]["names"]):
            per_call[f"{prefix}_{mode}"] = scores[:, j]
    f0_cols = [f"f0m_{m}" for m in bases["f0"]["names"]][:3]
    fam_x = pd.concat([calls[list(FAMILY_INPUT)], per_call[f0_cols]], axis=1).to_numpy(float)
    scaler = StandardScaler().fit(fam_x[disc])
    km = KMeans(N_FAMILIES, n_init=10, random_state=0).fit(scaler.transform(fam_x[disc]))
    families = km.predict(scaler.transform(fam_x))
    # Order columns so the first three f0 modes (by variance) drive the variability block.
    ordered = [f"f0m_{m}" for m in bases["f0"]["names"]]
    rest = [c for c in per_call.columns if c not in ordered]
    return per_call[ordered + rest], families, bases


def discover(
    d: pd.DataFrame, blocks: dict[str, list[str]], baseline: list[str], held_out: str
) -> dict[str, Any]:
    """Select age-trending features and predict decoding-gain direction on discovery litters."""
    disc = _discovery_rows(d, held_out)
    cands = [c for b in blocks.values() for c in b]
    trends = {c: age_trend(disc, c, litter_term=True) for c in cands}
    sig = bh([trends[c]["p"] for c in cands])
    selected = {
        c: int(np.sign(trends[c]["slope_sd_per_day"])) for c, s in zip(cands, sig, strict=True) if s
    }
    gains = {b: inner_gain(disc, baseline, cols) for b, cols in blocks.items()}
    gains["all_selected"] = inner_gain(disc, baseline, list(selected)) if selected else 0.0
    return {
        "held_out_litter": held_out,
        "discovery_litters": sorted(disc["litter_id"].unique()),
        "n_discovery_pup_days": int(len(disc)),
        "candidates": cands,
        "blocks": blocks,
        "baseline": baseline,
        "selected_features_predicted_sign": selected,
        "discovery_trends": trends,
        "predicted_gain_direction": {b: int(np.sign(g)) for b, g in gains.items()},
        "discovery_inner_gain_days": {b: round(g, 3) for b, g in gains.items()},
        "fdr_q": FDR_Q,
    }  # fmt: skip


def lock(pred: dict[str, Any], out: Path) -> tuple[Path, str]:
    path = out / f"m9_predictions_holdout_{pred['held_out_litter']}.json"
    write_json(path, pred)
    digest = sha256_file(path)
    with (out / "m9_locks.txt").open("a") as f:
        f.write(f"{digest}  {path.name}\n")
    return path, digest


def confirm(
    d: pd.DataFrame, locked_path: Path, digest: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Test the locked predictions on the held-out litter, without re-selection."""
    if hashlib.sha256(locked_path.read_bytes()).hexdigest() != digest:
        raise ValueError("locked predictions changed after locking")
    pred = json.loads(locked_path.read_text())
    litter = pred["held_out_litter"]
    held = d[d["litter_id"] == litter]
    train = d[d["litter_id"] != litter]
    feats = list(pred["selected_features_predicted_sign"])
    trends = {c: age_trend(held, c, litter_term=False) for c in feats}
    sig = bh([trends[c]["p"] for c in feats])
    rows = {}
    for c, s in zip(feats, sig, strict=True):
        sign = pred["selected_features_predicted_sign"][c]
        ok = bool(s and np.sign(trends[c]["slope_sd_per_day"]) == sign)
        rows[c] = trends[c] | {"predicted_sign": sign, "fdr_significant": s, "confirmed": ok}
    base = pred["baseline"]
    blocks = pred["blocks"] | {"all_selected": feats}
    s = held[["animal_id", "age"]].copy()
    s["p_base"] = fit_predict(train, held, base)
    mae_base = per_pup_mae(s, "p_base")
    gains = {}
    for b, cols in blocks.items():
        if not cols:
            continue
        s[f"p_{b}"] = fit_predict(train, held, base + cols)
        boot = paired_bootstrap(mae_base, per_pup_mae(s, f"p_{b}"), rng)
        direction = pred["predicted_gain_direction"][b]
        boot["predicted_direction"] = direction
        boot["confirmed"] = bool(
            (direction > 0 and boot["ci95"][0] > 0) or (direction < 0 and boot["ci95"][1] < 0)
        )
        gains[b] = boot
    return {"held_out_litter": litter, "locked_sha256": digest, "trend_tests": rows,
            "baseline_per_pup_mae": round(float(mae_base.mean()), 3), "gain_tests": gains,
            "weak_rotation": litter == WEAK_ROTATION}  # fmt: skip


def confirmation_table(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A finding is confirmed if it replicates in >= 2 of 3 rotations with the predicted sign."""
    feats = sorted({f for r in results for f in r["trend_tests"]})
    table = {}
    for f in feats:
        per = {r["held_out_litter"]: r["trend_tests"].get(f) for r in results}
        n_conf = sum(bool(v and v["confirmed"]) for v in per.values())
        signs = {v["predicted_sign"] for v in per.values() if v}
        table[f] = {"rotations_confirmed": n_conf, "confirmed": n_conf >= 2 and len(signs) == 1,
                    "sign": signs.pop() if len(signs) == 1 else None,
                    "per_rotation": {k: (None if v is None else {"slope": round(v["slope_sd_per_day"], 4),
                                     "p": v["p"], "confirmed": v["confirmed"]}) for k, v in per.items()}}  # fmt: skip
    blocks = sorted({b for r in results for b in r["gain_tests"]})
    gains = {}
    for b in blocks:
        per = {r["held_out_litter"]: r["gain_tests"].get(b) for r in results}
        n_conf = sum(bool(v and v["confirmed"]) for v in per.values())
        gains[b] = {"rotations_confirmed": n_conf, "confirmed": n_conf >= 2, "per_rotation": per}
    return {"features": table, "gains": gains}


def run_rotations(
    features_run: Path, controls_run: Path, out: Path, seed: int = 0
) -> dict[str, Any]:
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    load_splits(controls_run / "splits.json", digest)  # integrity check of the frozen split
    calls, curves = load_calls(features_run)
    day, baseline = load_days(controls_run, features_run)
    wp2 = beyond_contour(calls, curves)
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    bases_out: dict[str, Any] = {}
    for held_out in LITTERS:
        per_call, families, bases = per_call_table(calls, curves, wp2, held_out)
        cand, blocks = day_candidates(calls, per_call, families)
        d = day.merge(cand, left_on="wav_path", right_index=True, how="left")
        pred = discover(d, blocks, baseline, held_out)
        path, sha = lock(pred, out)
        results.append(confirm(d, path, sha, rng))  # first read of the held-out litter
        bases_out[held_out] = {k: {"names": b["names"], "var": [round(float(v), 4) for v in b["var"]],
                                   "template_cos": [round(v, 3) for v in b["template_cos"]]}
                               for k, b in bases.items()}  # fmt: skip
        arrays = {
            f"{k}_{p}": np.asarray(b[p]) for k, b in bases.items() for p in ("mean", "modes", "sd")
        }
        np.savez(out / f"m9_bases_holdout_{held_out}.npz", **arrays)  # type: ignore[arg-type]
        d.to_parquet(out / f"m9_day_features_holdout_{held_out}.parquet")
    table = confirmation_table(results)
    summary = {"rotations": results, "confirmation": table, "fpca": bases_out,
               "baseline": "classical + six H-M7a shape features (median, p90)",
               "n_calls": int(len(calls)), "n_pup_days": int(len(day))}  # fmt: skip
    write_json(out / "m9_confirmation.json", summary)
    return summary


# ---------------------------------------------------------------- WP6 biological contrasts

IDENTITY_FEATURES = ("log_duration", "f0_median_khz", "f0_min_khz", "f0_max_khz", "bandwidth_khz",
                     "slope_khz_per_s", "sinuosity", "tonality", "f0_start_khz", "f0_end_khz",
                     "onset_sweep_khz", "offset_sweep_khz", "u_index_khz", "curvature_khz",
                     "fm_residual_sd_khz", "fm_extrema_per_10ms")  # fmt: skip
AGE_BANDS = {"P1-7": (1, 7), "P8-14": (8, 14), "P15-21": (15, 21)}
CALLS_PER_PUP_DAY = 30


def sex_interactions(day_features: pd.DataFrame, confirmed: list[str]) -> dict[str, Any]:
    """Exploratory: age x sex on confirmed features, all litters, BH across features."""
    rows = {}
    for col in confirmed:
        sub = day_features.dropna(subset=[col])
        sub = sub.assign(y=(sub[col] - sub[col].mean()) / (sub[col].std() or 1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("y ~ age * C(sex) + C(litter_id)", sub, groups=sub["animal_id"]).fit(
                reml=True
            )
        term = next(k for k in m.params.index if k.startswith("age:"))
        rows[col] = {"term": term, "interaction_sd_per_day": round(float(m.params[term]), 4),
                     "p": float(m.pvalues[term]), "n_pups": int(sub["animal_id"].nunique())}  # fmt: skip
    sig = bh([r["p"] for r in rows.values()])
    for r, s in zip(rows.values(), sig, strict=True):
        r["fdr_significant"] = s
    return {"features": rows, "status": "exploratory; 28 pups, sex balance varies by litter"}


def identity_by_band(
    calls: pd.DataFrame, n_perm: int = 100, seed: int = 0
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Which pup made a call, within litter and age band, testing on unseen days.

    Leave-one-day-out within (litter, band); at most CALLS_PER_PUP_DAY calls per pup-day,
    pup-days with >= MIN_CALLS_VAR calls. Null: pup labels permuted among the pups recorded
    on the same day (keeps day structure). Score = balanced accuracy over pups.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(seed)
    c = calls.dropna(subset=["u_index_khz"]).copy()
    c["band"] = pd.cut(c["age"], [0, 7, 14, 21], labels=list(AGE_BANDS))
    sizes = c.groupby("wav_path")["age"].transform("size")
    c = c[sizes >= MIN_CALLS_VAR]
    c = c.groupby("wav_path", group_keys=False).apply(
        lambda d: d.sample(min(len(d), CALLS_PER_PUP_DAY), random_state=seed), include_groups=True
    )

    def score(d: pd.DataFrame, labels: np.ndarray) -> float:
        pred = np.empty(len(d), dtype=object)
        x = d[list(IDENTITY_FEATURES)].to_numpy(float)
        for a in d["age"].unique():
            test = (d["age"] == a).to_numpy()
            if len(set(labels[~test])) < 2:
                pred[test] = labels[test]
                continue
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LinearDiscriminantAnalysis())  # fmt: skip
            m.fit(x[~test], labels[~test])
            pred[test] = m.predict(x[test])
        return float(balanced_accuracy_score(labels, pred.astype(str)))

    rows = []
    for (litter, band), d in c.groupby(["litter_id", "band"], observed=True):
        d = d.reset_index(drop=True)
        if d["animal_id"].nunique() < 2 or d["age"].nunique() < 2:
            continue
        obs = score(d, d["animal_id"].to_numpy().astype(str))
        null = []
        for _ in range(n_perm):
            perm = d.groupby("age")["animal_id"].transform(
                lambda s: s.map(dict(zip(s.unique(), rng.permutation(s.unique()), strict=True)))
            )
            null.append(score(d, perm.to_numpy().astype(str)))
        rows.append({"litter": litter, "band": band, "pups": int(d["animal_id"].nunique()),
                     "days": int(d["age"].nunique()), "calls": int(len(d)),
                     "balanced_accuracy": round(obs, 3), "chance": round(1 / d["animal_id"].nunique(), 3),
                     "null_mean": round(float(np.mean(null)), 3),
                     "p_perm": (1 + sum(v >= obs for v in null)) / (1 + n_perm)})  # fmt: skip
    table = pd.DataFrame(rows)
    table["above_null"] = table["balanced_accuracy"] - table["null_mean"]
    return {
        "rows": table.to_dict(orient="records"),
        "status": "exploratory; sessions are confounded with pup within day",
    }, table


SEX_TABLE_ROTATION = "E2L2"  # bases fitted on the largest discovery set (E2L1 + E2L3, 24 pups)


def run_m9(features_run: Path, controls_run: Path, out: Path) -> dict[str, Any]:
    summary = run_rotations(features_run, controls_run, out)
    confirmed = [f for f, v in summary["confirmation"]["features"].items() if v["confirmed"]]
    day = pd.read_parquet(out / f"m9_day_features_holdout_{SEX_TABLE_ROTATION}.parquet")
    calls, _ = load_calls(features_run)
    identity, table = identity_by_band(calls)
    table.to_csv(out / "m9_identity_by_band.csv", index=False)
    extra = {
        "confirmed_features": confirmed,
        "sex_by_age": sex_interactions(day, confirmed) | {"table_rotation": SEX_TABLE_ROTATION},
        "identity_by_band": identity,
    }
    write_json(out / "m9_biology.json", extra)
    return summary | extra
