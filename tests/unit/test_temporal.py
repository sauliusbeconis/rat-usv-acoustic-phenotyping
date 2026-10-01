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
