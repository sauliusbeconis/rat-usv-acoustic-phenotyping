import numpy as np
import pandas as pd
import pytest

from rat_usv.signal.curves import N_POINTS, resample, runs
from rat_usv.signal.detect import DetectorConfig

FS = 214_285


def _frames(x):
    from rat_usv.signal.detect import whitened_spectrogram
    from rat_usv.signal.features import contour_frames

    cfg = DetectorConfig()
    f, t, db = whitened_spectrogram(x, FS, cfg)
    sel = (t >= 0.5) & (t <= 0.6)
    f_khz = (f / 1000).astype(np.float32)
    return cfg, f_khz, t[sel], db[:, sel], contour_frames(f_khz, t[sel], db[:, sel], cfg)


def _tone(*parts):
    rng = np.random.default_rng(0)
    t = np.arange(int(1.0 * FS)) / FS
    x = rng.normal(0, 0.01, t.size)
    m = (t >= 0.5) & (t < 0.6)
    for amp, freq in parts:
        x[m] += amp * np.sin(2 * np.pi * freq * t[m])
    return x.astype(np.float32)


def test_runs_counts_long_runs_only():
    assert runs(np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=bool)) == 2


def test_resample_ignores_nan_and_spans_series():
    t = np.linspace(0, 1, 11)
    v = t * 10
    v[3] = np.nan
    r = resample(t, v)
    assert r.shape == (N_POINTS,) and r[0] == 0 and r[-1] == pytest.approx(10)
    assert np.isnan(resample(t[:1], v[:1])).all()


def test_curves_harmonic_and_biphonation():
    from rat_usv.signal.curves import call_curves

    cfg, f_khz, t, db, fr = _frames(_tone((0.05, 40_000), (0.03, 80_000)))
    c = call_curves(f_khz, t, db, fr.times, fr.contour, fr.columns, cfg)
    assert 38 < np.median(c["f0"]) < 42
    assert np.nanmedian(c["harmonic_ratio"]) > -15  # harmonic present
    assert c["n_biphonation_events"] == 0  # 2*f0 is a harmonic, not a second voice
    cfg, f_khz, t, db, fr = _frames(_tone((0.05, 40_000), (0.04, 57_000)))
    c = call_curves(f_khz, t, db, fr.times, fr.contour, fr.columns, cfg)
    assert c["n_biphonation_events"] >= 1  # 57 kHz is not k * 20 kHz


def test_subharmonic_flag():
    from rat_usv.signal.curves import subharmonic_flags

    cfg, f_khz, t, db, fr = _frames(_tone((0.05, 60_000), (0.012, 30_000)))
    # f0 tracking picks the 30 kHz band (half rule), so test the flag directly at 60 kHz.
    flags = subharmonic_flags(db[:, fr.columns], f_khz, np.full(fr.columns.size, 60.0))
    assert flags.mean() > 0.5
    flags = subharmonic_flags(db[:, fr.columns], f_khz, np.full(fr.columns.size, 80.0))
    assert flags.mean() < 0.1


def test_fpca_names_modes_by_legendre_template():
    from rat_usv.evaluation.m9 import fpca, project

    rng = np.random.default_rng(0)
    tau = np.linspace(-1, 1, N_POINTS)
    x = (
        50
        + rng.normal(0, 3, (500, 1)) * tau
        + rng.normal(0, 2.5, (500, 1)) * (1.5 * tau**2 - 0.5)
        + rng.normal(0, 0.05, (500, N_POINTS))
    )
    b = fpca(x)
    assert b["names"][:2] == ["P1", "P2"] and b["template_cos"][0] > 0.99
    s = project(x, b)
    assert np.corrcoef(s[:, 0], x[:, -1] - x[:, 0])[0, 1] > 0.95  # sign follows the template


def test_confirmation_requires_two_rotations_same_sign():
    from rat_usv.evaluation.m9 import confirmation_table

    def rot(litter, ok, sign=1):
        t = {"slope_sd_per_day": 0.1 * sign, "p": 0.001, "predicted_sign": sign, "confirmed": ok}
        return {"held_out_litter": litter, "trend_tests": {"f": t}, "gain_tests": {}}

    assert confirmation_table([rot("A", True), rot("B", True), rot("C", False)])["features"]["f"][
        "confirmed"
    ]
    assert not confirmation_table([rot("A", True), rot("B", False), rot("C", False)])["features"][
        "f"
    ]["confirmed"]
    mixed = confirmation_table([rot("A", True), rot("B", True, -1), rot("C", False)])
    assert not mixed["features"]["f"]["confirmed"]


def test_discovery_guard_rejects_held_out_litter():
    from rat_usv.evaluation.m9 import _discovery_rows

    d = pd.DataFrame({"litter_id": ["A", "B", "C"]})
    assert set(_discovery_rows(d, "C")["litter_id"]) == {"A", "B"}
