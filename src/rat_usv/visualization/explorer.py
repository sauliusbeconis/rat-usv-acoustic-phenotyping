"""Exploratory latent-space explorer (visualisation class 2): click a call to see and hear it.

One held-out litter in its own fold's VAE space (fold spaces are not comparable). PCA is fitted
on the training litters' embeddings. Audio is the original WAV segment written at 1/16 of the
sample rate (16x slower, ~2.5 kHz for a 40 kHz call). Output is a single private HTML file.
"""

import base64
import io
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from PIL import Image

from rat_usv.representations import vae

SLOWDOWN = 16


def _png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).resize((192, 192)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _audio(path: Path, start: float, end: float) -> str:
    info = sf.info(str(path))
    fs = info.samplerate
    x, _ = sf.read(
        str(path),
        start=max(0, int((start - 0.02) * fs)),
        stop=int((end + 0.02) * fs),
        dtype="int16",
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs // SLOWDOWN)
        w.writeframes(x.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def build_explorer(
    source_root: Path,
    m4_run: Path,
    images_run: Path,
    explain_run: Path,
    litter: str,
    n: int,
    out: Path,
) -> dict[str, Any]:
    idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    images = np.asarray(np.load(images_run / "calls_images.npy", mmap_mode="r"))
    fam = pd.read_parquet(explain_run / "held_out_call_families.parquet")[
        ["wav_path", "call_index", "family"]
    ]
    idx = idx.merge(fam, on=["wav_path", "call_index"], how="left")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vae.VAE().to(device)
    model.load_state_dict(torch.load(m4_run / f"vae_holdout_{litter}.pt", map_location=device))
    z = vae.embed(model, images, device)
    train = (idx["litter_id"] != litter).to_numpy()
    mu = z[train].mean(0)
    _, _, vt = np.linalg.svd(z[train] - mu, full_matrices=False)
    xy = (z - mu) @ vt[:2].T
    held = np.flatnonzero(~train)
    rng = np.random.default_rng(0)
    ages = idx["postnatal_age_days"].to_numpy()
    pick = np.concatenate([rng.choice(held[ages[held] == a], min(n // 21 + 1, (ages[held] == a).sum()), replace=False)
                           for a in sorted(set(ages[held]))])[:n]  # fmt: skip
    points = []
    for i in pick:
        r = idx.iloc[i]
        points.append({
            "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
            "age": int(r["postnatal_age_days"]), "animal": r["animal_id"], "family": int(r["family"]),
            "dur": round((r["end_s"] - r["start_s"]) * 1000, 1), "wav": r["wav_path"],
            "t": round(float(r["start_s"]), 3), "img": _png(images[i]),
            "audio": _audio(source_root / r["wav_path"], r["start_s"], r["end_s"]),
        })  # fmt: skip
    (out / "explorer.html").write_text(
        PAGE.replace("__DATA__", json.dumps(points)).replace("__LITTER__", litter)
    )
    return {
        "points": len(points),
        "litter": litter,
        "slowdown": SLOWDOWN,
        "space": "fold VAE, PCA on training litters",
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>USV latent explorer</title>
<style>body{font-family:sans-serif;margin:0;display:flex;height:100vh}#plot{flex:1}#side{width:300px;padding:12px;border-left:1px solid #ccc;overflow:auto}
.warn{background:#fff3cd;padding:6px;font-size:12px}circle{cursor:pointer;stroke:#0003}circle:hover{stroke:#000;stroke-width:2}</style></head>
<body><svg id="plot"></svg><div id="side"><div class="warn"><b>Exploratory view, not evidence.</b> Held-out litter __LITTER__ in its fold's VAE space,
2-D PCA (fitted on training litters). Distances in 2-D can mislead; evidence is in decisions 0013/0014.</div>
<p>Colour = postnatal day (blue P1 → red P21). Click a point.</p><div id="info"></div></div>
<script>
const D=__DATA__;const svg=document.getElementById('plot');const W=svg.clientWidth,H=svg.clientHeight,pad=30;
const xs=D.map(d=>d.x),ys=D.map(d=>d.y);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
const sx=v=>pad+(v-x0)/(x1-x0)*(W-2*pad),sy=v=>H-pad-(v-y0)/(y1-y0)*(H-2*pad);
const col=a=>{const t=(a-1)/20;return `rgb(${Math.round(40+215*t)},${Math.round(90+40*(1-Math.abs(2*t-1)))},${Math.round(230-200*t)})`};
D.forEach((d,i)=>{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',sx(d.x));c.setAttribute('cy',sy(d.y));
c.setAttribute('r',5);c.setAttribute('fill',col(d.age));c.onclick=()=>show(d);svg.appendChild(c)});
function show(d){document.getElementById('info').innerHTML=`<b>P${d.age}</b> · ${d.animal} · family ${d.family}<br>${d.dur} ms · ${d.wav} @ ${d.t}s<br>
<img src="data:image/png;base64,${d.img}" width="256"><br><audio controls autoplay src="data:audio/wav;base64,${d.audio}"></audio><br><small>audio slowed 16×</small>`}
</script></body></html>"""
