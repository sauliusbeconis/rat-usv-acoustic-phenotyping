"""M3 negative controls on animal-day features with frozen animal-level splits.

Every decoder is a fixed, simple pipeline (median imputation + missingness indicators,
standardisation, ridge or logistic regression). Hyperparameters are tuned only by inner
grouped CV on training animals. These are controls, not candidate models.
"""

import warnings
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
EPOCH = date(2019, 12, 1)  # calendar-day origin for the date control
SNR_COLUMNS = ["median_peak_snr_db", "median_mean_snr_db", "bg_broadband_db"]
EXCLUDE_FROM_CLASSICAL = {"median_peak_snr_db", "median_mean_snr_db", "median_peak_freq_khz"}


def feature_sets(day: pd.DataFrame, bg_cols: list[str]) -> dict[str, list[str]]:
    classical = [
        c for c in day.columns if c.startswith("median_") and c not in EXCLUDE_FROM_CLASSICAL
    ] + ["log_call_rate", "log_ici_median", "ici_short_share"]
    return {"classical": classical, "background_only": bg_cols, "snr_amplitude_only": SNR_COLUMNS}


def add_derived(day: pd.DataFrame, birth: dict[str, date]) -> pd.DataFrame:
    d = day.copy()
    d["log_call_rate"] = np.log(d["call_rate_per_min"] + 0.1)
    d["log_ici_median"] = np.log(d["ici_median_s"])
    d["age"] = d["postnatal_age_days"].astype(float)
    d["date_index"] = [
        (birth[litter] - EPOCH).days + age
        for litter, age in zip(d["litter_id"], d["age"], strict=True)
    ]
    return d


def regressor() -> GridSearchCV:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    return GridSearchCV(
        pipe, {"ridge__alpha": ALPHAS}, cv=GroupKFold(5), scoring="neg_mean_absolute_error"
    )


def outer_predict(
    d: pd.DataFrame, cols: list[str], target: str, folds: list[dict[str, Any]]
) -> np.ndarray:
    """Out-of-fold predictions; each fold's model sees training animals only."""
    pred = np.full(len(d), np.nan)
    X, y, g = d[cols].to_numpy(float), d[target].to_numpy(float), d["animal_id"].to_numpy()
    for fold in folds:
        test = np.isin(g, fold["test_animals"])
        model = regressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[~test], y[~test], groups=g[~test])
        pred[test] = model.predict(X[test])
    return pred


def score(
    d: pd.DataFrame, pred: np.ndarray, target: str, rng: np.random.Generator
) -> dict[str, Any]:
    err = np.abs(pred - d[target].to_numpy(float))
    per_animal = pd.Series(err).groupby(d["animal_id"].to_numpy()).mean()
    boots = [per_animal.sample(frac=1, replace=True, random_state=int(rng.integers(1 << 31))).mean()
             for _ in range(1000)]  # fmt: skip
    rho = stats.spearmanr(pred, d[target]).statistic
    return {
        "mae_days": round(float(per_animal.mean()), 3),
        "mae_ci95": [
            round(float(np.percentile(boots, 2.5)), 3),
            round(float(np.percentile(boots, 97.5)), 3),
        ],
        "spearman_rho": round(float(rho), 3),
        "n_animals": int(per_animal.size),
        "n_animal_days": int(len(d)),
    }


def permutation_null(
    d: pd.DataFrame, cols: list[str], folds: list[dict[str, Any]], n: int, rng: np.random.Generator
) -> list[float]:
    """Age shuffled within each animal: keeps animal structure, breaks the feature-age link."""
    null = []
    for _ in range(n):
        p = d.copy()
        p["age"] = p.groupby("animal_id")["age"].transform(lambda s: rng.permutation(s.to_numpy()))
        pred = outer_predict(p, cols, "age", folds)
        null.append(
            float(
                pd.Series(np.abs(pred - p["age"])).groupby(p["animal_id"].to_numpy()).mean().mean()
            )
        )
    return null


def date_vs_age(d: pd.DataFrame, outcome: str) -> dict[str, Any]:
    """Mixed-model LRTs: does calendar date explain the outcome beyond age, and age beyond date?

    date_index = age + litter birth offset, so a litter fixed effect would make the two
    collinear. Litter is therefore omitted here: the date term is identified only through the
    three litters' birth offsets, so "date beyond age" cannot be separated from litter differences.
    """
    sub = d if outcome == "log_call_rate" else d[d["n_calls"] >= 5]
    sub = sub.dropna(subset=[outcome]).assign(y=lambda x: x[outcome])
    base = "C(sex)"
    age = 'cr(age, df=4, constraints="center")'
    dt = 'cr(date_index, df=3, constraints="center")'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def llf(rhs: str) -> tuple[float, int]:
            m = smf.mixedlm(f"y ~ {rhs}", sub, groups=sub["animal_id"]).fit(reml=False)
            return float(m.llf), len(m.fe_params)

        both, k_both = llf(f"{age} + {dt} + {base}")
        age_only, k_age = llf(f"{age} + {base}")
        date_only, k_date = llf(f"{dt} + {base}")
    date_chi2, age_chi2 = 2 * (both - age_only), 2 * (both - date_only)
    return {
        "n_animal_days": int(len(sub)),
        "date_given_age": {"chi2": round(date_chi2, 2), "df": k_both - k_age,
                           "p": float(stats.chi2.sf(date_chi2, k_both - k_age))},
        "age_given_date": {"chi2": round(age_chi2, 2), "df": k_both - k_date,
                           "p": float(stats.chi2.sf(age_chi2, k_both - k_date))},
    }  # fmt: skip


def identity_decoding(d: pd.DataFrame, cols: list[str], label: str) -> dict[str, Any]:
    """Train on even postnatal days, test on odd (and swap); fixed logistic regression."""
    X, y = d[cols].to_numpy(float), d[label].to_numpy()
    even = (d["age"] % 2 == 0).to_numpy()
    correct = total = 0
    for train in (even, ~even):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000),
        )
        model.fit(X[train], y[train])
        correct += int((model.predict(X[~train]) == y[~train]).sum())
        total += int((~train).sum())
    classes = d[label].nunique()
    return {
        "accuracy": round(correct / total, 3),
        "chance": round(1 / classes, 3),
        "classes": int(classes),
    }


def run_controls(
    day: pd.DataFrame,
    birth: dict[str, date],
    splits: dict[str, Any],
    bg_cols: list[str],
    n_permutations: int,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d = add_derived(day, birth)
    sets = feature_sets(d, bg_cols)
    loao, lolo = splits["leave_one_animal_out"], splits["leave_one_litter_out"]
    out: dict[str, Any] = {"feature_sets": sets, "age_decoding_loao": {}, "age_decoding_lolo": {}}
    for name, cols in sets.items():
        pred = outer_predict(d, cols, "age", loao)
        res = score(d, pred, "age", rng)
        null = permutation_null(d, cols, loao, n_permutations, rng)
        res["permutation_null_mae_mean"] = round(float(np.mean(null)), 3)
        res["permutation_p"] = (1 + sum(v <= res["mae_days"] for v in null)) / (1 + len(null))
        out["age_decoding_loao"][name] = res
        lolo_pred = outer_predict(d, cols, "age", lolo)
        lres = score(d, lolo_pred, "age", rng)
        lres["bias_by_held_out_litter"] = {
            litter: round(float((lolo_pred[m] - d["age"].to_numpy()[m]).mean()), 2)
            for litter in sorted(d["litter_id"].unique())
            for m in [(d["litter_id"] == litter).to_numpy()]
        }
        out["age_decoding_lolo"][name] = lres
    out["date_vs_age"] = {
        o: date_vs_age(d, o)
        for o in ("log_call_rate", "median_f0_median_khz", "median_duration_ms", "median_tonality")
    }
    out["identity_decoding"] = {
        "animal_all": identity_decoding(d, sets["classical"], "animal_id"),
        "litter": identity_decoding(d, sets["classical"], "litter_id"),
        **{
            f"animal_within_{litter}": identity_decoding(g, sets["classical"], "animal_id")
            for litter, g in d.groupby("litter_id")
        },
    }
    out["birth_dates"] = {k: v.isoformat() for k, v in birth.items()}
    return out
