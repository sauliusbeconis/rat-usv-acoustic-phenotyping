from datetime import date

import numpy as np
import pandas as pd
import pytest

from rat_usv.evaluation.controls import identity_decoding, outer_predict, score
from rat_usv.evaluation.splits import load_splits, make_splits, write_splits


def animals():
    return {f"E9L{i % 3 + 1}-{i:04d}": f"E9L{i % 3 + 1}" for i in range(9)}


def test_splits_cover_every_animal_once(tmp_path):
    splits = make_splits(animals())
    assert len(splits["leave_one_animal_out"]) == 9 and len(splits["leave_one_litter_out"]) == 3
    digest = write_splits(splits, tmp_path / "splits.json")
    assert load_splits(tmp_path / "splits.json", digest)["animals"] == animals()
    with pytest.raises(ValueError, match="digest"):
        load_splits(tmp_path / "splits.json", "0" * 64)


def days(signal: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for animal, litter in animals().items():
        for age in range(1, 22):
            rows.append({"animal_id": animal, "litter_id": litter, "age": float(age),
                         "x1": signal * age + rng.normal(0, 1), "x2": rng.normal(0, 1)})  # fmt: skip
    return pd.DataFrame(rows)


def test_outer_predict_recovers_age_only_when_features_carry_it():
    folds = make_splits(animals())["leave_one_animal_out"]
    rng = np.random.default_rng(0)
    informative = days(1.0)
    good = score(informative, outer_predict(informative, ["x1", "x2"], "age", folds), "age", rng)
    noise = days(0.0)
    bad = score(noise, outer_predict(noise, ["x1", "x2"], "age", folds), "age", rng)
    assert good["mae_days"] < 1.5 and good["spearman_rho"] > 0.9
    assert bad["mae_days"] > 4.0


def test_identity_decoding_near_chance_on_noise():
    result = identity_decoding(days(0.0, seed=3), ["x2"], "animal_id")
    assert result["classes"] == 9 and result["accuracy"] < 0.3


def test_epoch_is_before_all_births():
    from rat_usv.evaluation.controls import EPOCH

    assert EPOCH <= date(2019, 12, 1)
