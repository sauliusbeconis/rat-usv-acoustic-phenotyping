"""M1 hand-check: blind review page of sampled events, and scoring of the owner's labels."""

import base64
import html
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy import signal

from rat_usv.config import SourceConfig
from rat_usv.data.avisoft import load_references
from rat_usv.data.linkage import final_pup_recordings, load_recordings
from rat_usv.signal.detect import Call
from rat_usv.signal.evaluate import match_calls
from rat_usv.signal.pipeline import AGE_BANDS, ARCHIVE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUOTA = {"detector_only": 100, "avisoft_only": 50, "agreed": 50, "empty": 40}
CONTEXT_S = 0.15


def _band(age: int) -> str:
    return next(f"P{lo}-P{hi}" for lo, hi in AGE_BANDS if lo <= age <= hi)


def sample_events(cfg: SourceConfig, detect_run: Path, seed: int = 7) -> list[dict[str, Any]]:
    """Stratified (litter x age band) sample of events from non-tuning recordings."""
    assert cfg.linkage_dir is not None
    summary = json.loads((detect_run / "detection_summary.json").read_text())
    tuning = set(summary["tuning_recordings"])
    recs = {r.wav_path: r for r in final_pup_recordings(load_recordings(cfg.linkage_dir))}
    refs = load_references(cfg.source_root / ARCHIVE, list(recs.values()))
    table = pq.read_table(detect_run / "calls.parquet").to_pylist()
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        calls[row["wav_path"]].append(row)
    background = defaultdict(list)
    for row in pq.read_table(detect_run / "background_windows.parquet").to_pylist():
        background[row["wav_path"]].append(row)

    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path, rec in recs.items():
        if path in tuning or path not in refs or rec.postnatal_age_days is None:
            continue
        stratum = f"{rec.litter_id}|{_band(rec.postnatal_age_days)}"
        det = [
            Call(**{k: v for k, v in c.items() if k in Call.__dataclass_fields__})
            for c in calls[path]
        ]
        m = match_calls(det, refs[path])
        used_d = {i for i, _ in m.pairs}
        used_r = {j for _, j in m.pairs}
        base = {"wav_path": path, "stratum": stratum, "animal_id": rec.animal_id}
        for i, c in enumerate(det):
            kind = "agreed" if i in used_d else "detector_only"
            pools[kind][stratum].append(
                {**base, "kind": kind, "start_s": c.start_s, "end_s": c.end_s}
            )
        for j, q in enumerate(refs[path]):
            if j not in used_r and q.end_s > q.start_s:
                pools["avisoft_only"][stratum].append(
                    {**base, "kind": "avisoft_only", "start_s": q.start_s, "end_s": q.end_s}
                )
        for w in background[path]:
            pools["empty"][stratum].append(
                {**base, "kind": "empty", "start_s": w["start_s"], "end_s": w["end_s"]}
            )

    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    for kind, quota in QUOTA.items():
        strata = sorted(pools[kind])
        per = max(1, quota // max(1, len(strata)))
        for stratum in strata:
            pool = pools[kind][stratum]
            for e in rng.sample(pool, min(per, len(pool))):
                events.append({**e, "pool_size": len(pool)})
    rng.shuffle(events)
    for n, e in enumerate(events):
        e["item"] = f"item-{n:03d}"
    return events


def _image(path: Path, start: float, end: float) -> str:
    info = sf.info(str(path))
    fs = info.samplerate
    a = max(0.0, start - CONTEXT_S)
    b = min(info.duration, end + CONTEXT_S)
    x, _ = sf.read(str(path), start=int(a * fs), stop=int(b * fs), dtype="float32")
    f, t, p = signal.spectrogram(x, fs, nperseg=512, noverlap=384)
    db = 10 * np.log10(p + 1e-20)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(
        t + a, f / 1000, db, shading="auto", cmap="magma",
        vmin=np.percentile(db, 75), vmax=np.percentile(db, 99.9),
    )  # fmt: skip
    ax.axvspan(start, end, edgecolor="cyan", facecolor="none", linewidth=1.5)
    ax.set_ylim(0, 107)
    ax.set_xlabel("s")
    ax.set_ylabel("kHz")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>USV detection check</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}}
.item{{border:1px solid #ccc;margin:12px 0;padding:8px}} .item.done{{background:#eef8ee}}
button{{margin:4px;padding:6px 12px}} button.sel{{background:#333;color:#fff}}
#bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ccc}}</style></head>
<body><div id="bar"><b>Is there a USV call inside the cyan box?</b> Answered: <span id="n">0</span>/{total}
<button onclick="save()">Download labels</button></div>
<p>Each spectrogram shows {context} s of context on both sides. The box may contain a call, part of a call,
noise or nothing. Judge only what is inside the box. Answers are kept in this browser until you download them.</p>
{items}
<script>
const KEY="usv-check-{run}";let L={{}};try{{L=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
function mark(id,v){{L[id]=v;try{{localStorage.setItem(KEY,JSON.stringify(L))}}catch(e){{}};draw()}}
function draw(){{document.querySelectorAll('.item').forEach(d=>{{const v=L[d.id];d.classList.toggle('done',!!v);
d.querySelectorAll('button').forEach(b=>b.classList.toggle('sel',b.dataset.v===v))}});
document.getElementById('n').textContent=Object.keys(L).length}}
function save(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(L,null,1)],{{type:'application/json'}}));
a.download='labels.json';a.click()}}
draw();</script></body></html>"""


def build_review(cfg: SourceConfig, detect_run: Path, out: Path) -> int:
    events = sample_events(cfg, detect_run)
    blocks = []
    for e in events:
        img = _image(cfg.source_root / e["wav_path"], e["start_s"], e["end_s"])
        buttons = "".join(
            f"<button data-v=\"{v}\" onclick=\"mark('{e['item']}','{v}')\">{label}</button>"
            for v, label in (("call", "Call"), ("not_call", "Not a call"), ("unsure", "Unsure"))
        )
        blocks.append(
            f'<div class="item" id="{e["item"]}"><div>{html.escape(e["item"])}</div>'
            f'<img src="data:image/png;base64,{img}" width="780"><div>{buttons}</div></div>'
        )
    (out / "review.html").write_text(
        PAGE.format(total=len(events), context=CONTEXT_S, items="\n".join(blocks), run=out.name)
    )
    (out / "review_items.json").write_text(json.dumps(events, indent=1))
    return len(events)


def score_review(items: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    """Share of real calls per event kind (unsure excluded), with Wilson 95% intervals."""

    def wilson(k: int, n: int) -> tuple[float | None, float | None]:
        if n == 0:
            return None, None
        p, z = k / n, 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return round(float(centre - half), 3), round(float(centre + half), 3)

    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for e in items:
        by_kind[e["kind"]][labels.get(e["item"], "missing")] += 1
    out: dict[str, Any] = {}
    for kind, c in sorted(by_kind.items()):
        n = c["call"] + c["not_call"]
        out[kind] = {
            **dict(c),
            "share_real_call": round(c["call"] / n, 3) if n else None,
            "ci95": wilson(c["call"], n),
        }
    return out
