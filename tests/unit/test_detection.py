import io
import zipfile

import numpy as np
import pytest
import soundfile as sf

from rat_usv.data.avisoft import parse_export
from rat_usv.signal.detect import DetectorConfig, detect_array, detect_file
from rat_usv.signal.evaluate import match_calls
from rat_usv.signal.pipeline import background_windows

FS = 214_285


def synthetic(seconds=2.0, calls=((0.5, 0.56, 40_000), (1.2, 1.25, 60_000)), click_at=None):
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * FS)) / FS
    x = rng.normal(0, 0.01, t.size)
    for start, end, freq in calls:
        m = (t >= start) & (t < end)
        inst = freq + 2000 * (t[m] - start) / (end - start)  # 2 kHz upward sweep
        x[m] += 0.05 * np.sin(2 * np.pi * np.cumsum(inst) / FS)
    if click_at is not None:
        i = int(click_at * FS)
        x[i : i + 40] += rng.normal(0, 0.5, 40)
    return x.astype(np.float32)


def test_detects_tonal_calls_and_rejects_click():
    calls = detect_array(synthetic(click_at=1.6), FS, DetectorConfig())
    assert len(calls) == 2
    assert abs(calls[0].start_s - 0.5) < 0.005 and abs(calls[0].end_s - 0.56) < 0.005
    assert 39 < calls[0].peak_freq_khz < 43 and 59 < calls[1].peak_freq_khz < 63


def test_low_frequency_tone_rejected_by_peak_floor():
    calls = detect_array(synthetic(calls=((0.5, 0.6, 21_000),)), FS, DetectorConfig())
    assert calls == []


def test_chunked_file_matches_single_pass(tmp_path):
    x = synthetic(seconds=4.0, calls=((0.5, 0.56, 40_000), (1.99, 2.03, 45_000)))
    path = tmp_path / "x.wav"
    sf.write(path, x, FS, subtype="PCM_16")
    whole = detect_array(sf.read(path, dtype="float32")[0], FS, DetectorConfig())
    chunked, duration = detect_file(path, DetectorConfig(chunk_s=2.0, chunk_overlap_s=0.2))
    assert duration == pytest.approx(4.0, abs=1e-3)
    assert [round(c.start_s, 2) for c in chunked] == [round(c.start_s, 2) for c in whole]


def test_match_calls_one_to_one():
    class C:
        def __init__(self, a, b):
            self.start_s, self.end_s = a, b

    detected = [C(0.0, 1.0), C(2.0, 3.0), C(5.0, 6.0)]
    reference = [C(0.5, 1.5), C(0.9, 1.1), C(2.5, 2.6)]
    m = match_calls(detected, reference)
    assert (m.true_positive, m.false_positive, m.false_negative) == (2, 1, 1)
    assert m.precision == pytest.approx(2 / 3) and m.recall == pytest.approx(2 / 3)


def test_background_windows_avoid_calls():
    class C:
        start_s, end_s = 1.0, 1.1

    windows = background_windows([C()], 10.0, 5, 0.5, 0.2, seed=0)
    assert len(windows) == 5
    assert all(b <= 0.8 or a >= 1.3 for a, b in windows)


def _xlsx(rows):
    cells = "".join(
        f'<row r="{i}">'
        + "".join(
            f'<c r="{col}{i}" t="inlineStr"><is><t>{v}</t></is></c>'
            if isinstance(v, str)
            else f'<c r="{col}{i}"><v>{v}</v></c>'
            for col, v in zip("ABCD", row, strict=False)
        )
        + "</row>"
        for i, row in enumerate(rows, start=1)
    )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{cells}</sheetData></worksheet>"
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as book:
        book.writestr("xl/worksheets/sheet1.xml", sheet)
    return stream.getvalue()


def test_parse_avisoft_export_skips_summary_rows():
    data = _xlsx(
        [
            ["#", "class", "start time", "end time"],
            ["min", " ", 100, 200],
            [1, "Vidutines", 4890.3, 4950.7],
            [2, "Trumpos", 5073.1, 5080.0],
        ]
    )
    calls = parse_export(data)
    assert [(c.start_s, c.label) for c in calls] == [(4.8903, "Vidutines"), (5.0731, "Trumpos")]


def test_score_review_shares_and_intervals():
    from rat_usv.signal.review import score_review

    items = [{"item": f"i{n}", "kind": "detector_only"} for n in range(4)]
    items.append({"item": "e0", "kind": "empty"})
    labels = {"i0": "call", "i1": "call", "i2": "not_call", "i3": "unsure"}
    result = score_review(items, labels)
    assert result["detector_only"]["share_real_call"] == pytest.approx(2 / 3, abs=1e-3)
    low, high = result["detector_only"]["ci95"]
    assert low < 2 / 3 < high
    assert result["empty"]["missing"] == 1 and result["empty"]["share_real_call"] is None
