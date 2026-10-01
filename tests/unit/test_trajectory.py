import numpy as np
import pandas as pd

from rat_usv.statistics.trajectory import fit, population_curve, prepare


def synthetic_days(age_effect: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(12):
        litter = f"E9L{a % 3 + 1}"
        offset = rng.normal(0, 1)
        for age in range(1, 22):
            rows.append(
                {
                    "animal_id": f"{litter}-{a:04d}",
                    "litter_id": litter,
                    "sex": "female" if a % 2 else "male",
                    "postnatal_age_days": age,
                    "n_calls": 50,
                    "median_f0_median_khz": 40 + age_effect * age + offset + rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def test_age_effect_detected_and_curve_tracks_it():
    d = prepare(synthetic_days(0.8), "median_f0_median_khz")
    r = fit(d)
    assert r["age_lrt_p"] < 1e-6 and r["n_animals"] == 12
    curve = population_curve(r["model"], d)
    slope = np.polyfit(curve["age"], curve["mean"], 1)[0]
    assert 0.6 < slope < 1.0
    assert (curve["low"] < curve["mean"]).all() and (curve["mean"] < curve["high"]).all()


def test_no_age_effect_not_significant():
    r = fit(prepare(synthetic_days(0.0, seed=1), "median_f0_median_khz"))
    assert r["age_lrt_p"] > 0.01
