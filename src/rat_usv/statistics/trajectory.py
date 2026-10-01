"""M2 baseline: mixed-model developmental trajectories of classical animal-day features."""

import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix
from scipy import stats

from rat_usv.provenance.io import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTCOMES = {
    "call_rate_per_min": "Call rate (calls/min)",
    "median_f0_median_khz": "Median fundamental (kHz)",
    "median_duration_ms": "Median duration (ms)",
    "median_bandwidth_khz": "Median f0 bandwidth (kHz)",
    "median_slope_khz_per_s": "Median f0 slope (kHz/s)",
    "median_sinuosity": "Median sinuosity",
    "median_harmonic_share": "Median harmonic share",
    "median_tonality": "Median tonality",
    "ici_median_s": "Median inter-call interval (s)",
}
LOG_OUTCOMES = {"call_rate_per_min", "median_duration_ms", "ici_median_s"}
LOG_OFFSET = {"call_rate_per_min": 0.1}  # zero-call animal-days stay on the log axis
LITTER_COLOURS = {"E2L1": "#1f77b4", "E2L2": "#d62728", "E2L3": "#2ca02c"}
RHS = 'cr(age, df=4, constraints="center") + C(sex) + C(litter_id)'
MIN_CALLS = 5  # animal-days with fewer confident calls have no morphology medians in the fit


def prepare(day: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = day.copy()
    if outcome == "median_harmonic_share":
        d = d[d["n_harmonic_observable_calls"] >= MIN_CALLS]
    elif outcome != "call_rate_per_min":
        d = d[d["n_calls"] >= MIN_CALLS]
    d = d.dropna(subset=[outcome])
    d["y"] = (
        np.log(d[outcome] + LOG_OFFSET.get(outcome, 1e-3))
        if outcome in LOG_OUTCOMES
        else d[outcome]
    )
    d["age"] = d["postnatal_age_days"].astype(float)
    return d


def fit(d: pd.DataFrame) -> dict[str, Any]:
    """Random-intercept model y ~ spline(age, 4 df) + sex + litter; LRT for the age terms."""
    full_f = "y ~ " + RHS
    null_f = "y ~ C(sex) + C(litter_id)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = smf.mixedlm(full_f, d, groups=d["animal_id"]).fit(reml=False)
        null = smf.mixedlm(null_f, d, groups=d["animal_id"]).fit(reml=False)
        reml = smf.mixedlm(full_f, d, groups=d["animal_id"]).fit(reml=True)
    lr = 2 * (full.llf - null.llf)
    df_age = len(full.fe_params) - len(null.fe_params)
    var_animal = float(reml.cov_re.iloc[0, 0])
    var_resid = float(reml.scale)
    return {
        "model": reml,
        "n_animal_days": int(len(d)),
        "n_animals": int(d["animal_id"].nunique()),
        "age_lrt_chi2": float(lr),
        "age_lrt_df": int(df_age),
        "age_lrt_p": float(stats.chi2.sf(lr, df_age)),
        "animal_icc": var_animal / (var_animal + var_resid) if var_animal + var_resid else None,
        "converged": bool(reml.converged),
    }


def population_curve(model: Any, d: pd.DataFrame) -> pd.DataFrame:
    """Predicted mean over ages (sex/litter averaged by design proportions) with 95% CI."""
    ages = np.arange(int(d["age"].min()), int(d["age"].max()) + 1, dtype=float)
    rows = []
    fe = model.fe_params
    cov = model.cov_params().loc[fe.index, fe.index]
    design = dmatrix(RHS, d).design_info
    if list(design.column_names) != list(fe.index):
        raise ValueError("Design columns do not match fitted fixed effects")
    for age in ages:
        grid = d.assign(age=age)
        x = np.asarray(build_design_matrices([design], grid)[0]).mean(axis=0)
        mean = float(x @ fe.values)
        se = float(np.sqrt(x @ cov.values @ x))
        rows.append({"age": age, "mean": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se})
    return pd.DataFrame(rows)


def figure(d: pd.DataFrame, curve: pd.DataFrame, outcome: str, path: Path, note: str) -> None:
    offset = LOG_OFFSET.get(outcome, 1e-3)
    # Log outcomes are drawn as value + offset so zero-call days stay visible.
    back = (lambda v: np.exp(v)) if outcome in LOG_OUTCOMES else (lambda v: v)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for _animal, g in d.sort_values("age").groupby("animal_id"):
        ax.plot(g["age"], back(g["y"]), color=LITTER_COLOURS.get(g["litter_id"].iat[0], "grey"),
                alpha=0.35, lw=0.8)  # fmt: skip
    ax.fill_between(
        curve["age"], back(curve["low"]), back(curve["high"]), color="black", alpha=0.15
    )
    ax.plot(curve["age"], back(curve["mean"]), color="black", lw=2, label="population (95% CI)")
    for litter, colour in LITTER_COLOURS.items():
        ax.plot([], [], color=colour, label=litter)
    ax.set_xlabel("Postnatal day (P0 = birth)")
    label = OUTCOMES[outcome]
    ax.set_ylabel(f"{label} + {offset:g}" if outcome in LOG_OUTCOMES and offset >= 0.01 else label)
    if outcome in LOG_OUTCOMES:
        ax.set_yscale("log")
    ax.set_title(note, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_baseline(features_run: Path, out: Path) -> dict[str, Any]:
    day = pd.read_parquet(features_run / "animal_day.parquet")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "primary": day,
        "no_edit_suspected": day[~day["edit_suspected"]],
        "all_detections_rate": day.assign(call_rate_per_min=day["call_rate_all_per_min"]),
    }
    results: dict[str, Any] = {}
    curves = []
    for variant, data in variants.items():
        outcomes = ["call_rate_per_min"] if variant == "all_detections_rate" else list(OUTCOMES)
        for outcome in outcomes:
            d = prepare(data, outcome)
            try:
                r = fit(d)
            except Exception as error:  # record failures, never drop them silently
                results[f"{variant}|{outcome}"] = {"error": f"{type(error).__name__}: {error}"}
                continue
            curve = population_curve(r.pop("model"), d)
            results[f"{variant}|{outcome}"] = {"log_scale": outcome in LOG_OUTCOMES, **r}
            curves.append(curve.assign(variant=variant, outcome=outcome))
            if variant == "primary":
                note = (
                    f"{r['n_animals']} animals, {r['n_animal_days']} animal-days; age LRT "
                    f"chi2={r['age_lrt_chi2']:.1f}, df={r['age_lrt_df']}, p={r['age_lrt_p']:.2g}; "
                    f"animal ICC={r['animal_icc']:.2f}. Exploratory until M3 controls."
                )
                figure(d, curve, outcome, fig_dir / f"{outcome}.png", note)
    pd.concat(curves).to_csv(out / "population_curves.csv", index=False)
    summary = {
        "features_run": str(features_run),
        "model": "MixedLM: y ~ cr(age, df=4) + sex + litter, random intercept per animal; "
        "age tested by ML likelihood-ratio test; curves from REML fit",
        "min_confident_calls_per_animal_day": MIN_CALLS,
        "log_outcomes": sorted(LOG_OUTCOMES),
        "results": results,
        "status": "OBSERVED (exploratory baseline; M3 negative controls pending)",
    }
    write_json(out / "baseline_summary.json", summary)
    return summary
