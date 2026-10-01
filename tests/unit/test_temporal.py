import numpy as np
import pytest

pytest.importorskip("torch")

from rat_usv.evaluation.temporal import K, sequence_features, timing_features  # noqa: E402


def test_bouts_split_on_long_gaps():
    starts = np.array([0.0, 0.1, 0.2, 2.0, 2.1])
    ends = starts + 0.05
    f = timing_features(starts, ends, minutes=1.0)
    assert f["bouts_per_min"] == 2 and f["calls_per_bout"] == 2.5
    assert f["ici_q50"] == pytest.approx(0.1)


def test_sequence_entropy_zero_for_deterministic_order():
    fam = np.array([0, 1, 0, 1, 0, 1, 0])
    f = sequence_features(fam, np.arange(7, dtype=float))
    assert f["seq_entropy"] == pytest.approx(0.0) and f["repeat_rate"] == 0.0
    assert sum(f[f"t_{a}{b}"] for a in range(K) for b in range(K)) == pytest.approx(1.0)
    assert f["ici_lv"] == pytest.approx(0.0)  # perfectly regular rhythm


def test_too_few_calls_is_missing_not_zero():
    f = timing_features(np.array([0.0, 1.0]), np.array([0.1, 1.1]), 1.0)
    assert all(np.isnan(v) for v in f.values())


def test_fragile_x_table_aggregates_per_file_and_drops_small_files(tmp_path):
    import pandas as pd

    from rat_usv.evaluation.shape_test import fragile_x_table

    rows = []
    for f, rat, pnd, n in (
        ("a.mat", "r1", "p3", 6),
        ("b.mat", "r1", "p6", 6),
        ("c.mat", "r2", "p3", 2),
    ):
        for i in range(n):
            rows.append({"File": f, "Rat": rat, "PND": pnd, "Genotype": "WT", "Label": "Trill" if i % 2 else "Flat",
                         "Call Length (s)": 0.05, "Principal Frequency (kHz)": 40.0, "Delta Freq (kHz)": 3.0,
                         "Slope (kHz/s)": 1.0, "Peak Freq (kHz)": 41.0, "Sinuosity": 1.0 + i,
                         "Frequency Standard Deviation (kHz)": 0.5})  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp_path / "fx.csv", index=False)
    t = fragile_x_table(tmp_path / "fx.csv")
    assert sorted(t.index) == ["a.mat", "b.mat"]  # c.mat has < 5 calls
    assert t.loc["a.mat", "age"] == 3 and t.loc["a.mat", "fm_share_Trill"] == 0.5
    assert t.loc["a.mat", "fm_med_Sinuosity"] == 3.5
