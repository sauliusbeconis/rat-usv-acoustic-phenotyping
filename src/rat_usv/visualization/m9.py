"""M9 figures, drawn from a finished `usv m9` run directory (no refitting)."""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LITTERS = ("E2L1", "E2L2", "E2L3")
LITTER_COLOURS = {"E2L1": "#2a6fdb", "E2L2": "#d1495b", "E2L3": "#2a9d5c"}
BLOCK_LABELS = {"f0_modes": "f0 shape modes (fPCA)", "env_modes": "Loudness-envelope modes",
                "beyond": "Beyond-contour features", "variability": "Call-to-call variability",
                "all_selected": "All selected features"}  # fmt: skip


def label(f: str) -> str:
    return (f.removesuffix("_med").replace("f0m_", "f0 mode ").replace("envm_", "envelope mode ")
            .replace("var_within_family_", "within-family variability of ").replace("var_", "variability of ")
            .replace("share_", "share of calls with ").replace("_", " "))  # fmt: skip


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def modes_figure(run: Path, which: str, out: Path, rotation: str = "E2L2") -> None:
    """Mean curve +/- 2 SD of each mode (one rotation), and mode shapes across rotations."""
    summary = json.loads((run / "m9_confirmation.json").read_text())
    names = summary["fpca"][rotation][which]["names"]
    var = summary["fpca"][rotation][which]["var"]
    with np.load(run / f"m9_bases_holdout_{rotation}.npz") as z:
        mean, modes, sd = z[f"{which}_mean"], z[f"{which}_modes"], z[f"{which}_sd"]
    k = len(names)
    tau = np.linspace(0, 100, mean.size)
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 4.4), squeeze=False)
    unit = "kHz" if which == "f0" else "dB"
    for j in range(k):
        ax = axes[0, j]
        amp = 2 * sd[j]  # mean +/- 2 SD of the mode score
        ax.plot(tau, mean, color="0.3", lw=1.5, label="mean")
        ax.plot(tau, mean + amp * modes[j], color="#d1495b", lw=1.2, label="+2 SD")
        ax.plot(tau, mean - amp * modes[j], color="#2a6fdb", lw=1.2, label="−2 SD")
        ax.set_title(f"{names[j]} ({100 * var[j]:.0f}%)", fontsize=9)
        ax.set_xlabel("% of call", fontsize=8)
        if j == 0:
            ax.set_ylabel(
                f"{'f0' if which == 'f0' else 'level'} minus call mean ({unit})", fontsize=8
            )
            ax.legend(fontsize=6, frameon=False)
        ax2 = axes[1, j]
        for rot in LITTERS:
            names_r = summary["fpca"][rot][which]["names"]
            if names[j] not in names_r:
                continue
            with np.load(run / f"m9_bases_holdout_{rot}.npz") as z:
                ax2.plot(tau, z[f"{which}_modes"][names_r.index(names[j])], color=LITTER_COLOURS[rot],
                         lw=1.2, label=f"fit without {rot}")  # fmt: skip
        ax2.axhline(0, color="0.8", lw=0.6)
        ax2.set_xlabel("% of call", fontsize=8)
        if j == 0:
            ax2.set_ylabel("mode shape", fontsize=8)
            ax2.legend(fontsize=6, frameon=False)
    fig.suptitle(f"{'Fundamental-frequency' if which == 'f0' else 'Loudness-envelope'} shape modes "
                 "(top: one rotation; bottom: same mode refitted in each rotation)", fontsize=10)  # fmt: skip
    _save(fig, out)


def confirmation_heatmap(run: Path, out: Path) -> None:
    summary = json.loads((run / "m9_confirmation.json").read_text())
    table = summary["confirmation"]["features"]
    feats = sorted(table, key=lambda f: (-table[f]["rotations_confirmed"], f))
    mat = np.full((len(feats), 3), np.nan)
    status = np.zeros((len(feats), 3), dtype=int)  # 0 not selected, 1 tested, 2 confirmed
    for i, f in enumerate(feats):
        for j, lit in enumerate(LITTERS):
            v = table[f]["per_rotation"].get(lit)
            if v:
                mat[i, j] = v["slope"]
                status[i, j] = 2 if v["confirmed"] else 1
    lim = np.nanmax(np.abs(mat)) or 1
    fig, ax = plt.subplots(figsize=(7.4, 0.28 * len(feats) + 1.6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), s in np.ndenumerate(status):
        ax.text(j, i, {0: "–", 1: "·", 2: "✓"}[s], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3), [f"held-out {x}" for x in LITTERS], fontsize=8)
    ax.set_yticks(range(len(feats)),
                  [("★ " if table[f]["confirmed"] else "") + label(f) for f in feats], fontsize=7)  # fmt: skip
    fig.colorbar(im, label="age slope in held-out litter (SD per day)")
    ax.set_title(
        "Locked predictions tested on unseen litters\n✓ confirmed, · tested not confirmed, – not selected; ★ ≥ 2 of 3",
        fontsize=9,
    )
    _save(fig, out)


def gain_forest(run: Path, out: Path) -> None:
    summary = json.loads((run / "m9_confirmation.json").read_text())
    gains = summary["confirmation"]["gains"]
    blocks = [b for b in BLOCK_LABELS if b in gains]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(blocks) * 3 / 2 + 1.2))
    y = 0.0
    ticks, labels = [], []
    for b in blocks:
        for j, lit in enumerate(LITTERS):
            v = gains[b]["per_rotation"].get(lit)
            if not v:
                continue
            lo, hi = v["ci95"]
            colour = LITTER_COLOURS[lit]
            ax.plot([lo, hi], [y - j * 0.25] * 2, color=colour, lw=2)
            ax.plot(v["mean_mae_difference_days"], y - j * 0.25, "o", color=colour,
                    mfc="white" if not v["confirmed"] else colour, label=lit if b == blocks[0] else None)  # fmt: skip
        ticks.append(y - 0.25)
        labels.append(BLOCK_LABELS[b] + (" ★" if gains[b]["confirmed"] else ""))
        y -= 1.2
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_xlabel("age error saved per pup vs baseline (days; > 0 = feature block helps)")
    ax.set_title(
        "Added information beyond classical + H-M7a shape features\n(filled = CI excludes 0 in the predicted direction; ★ ≥ 2 of 3 litters)",
        fontsize=9,
    )
    ax.legend(title="held-out litter", fontsize=7, title_fontsize=7, frameon=False)
    _save(fig, out)


def trend_panels(run: Path, feats: list[str], out: Path, rotation: str = "E2L2") -> None:
    if not feats:
        return
    day = pd.read_parquet(run / f"m9_day_features_holdout_{rotation}.parquet")
    n = len(feats)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows), squeeze=False)
    for ax, f in zip(axes.ravel(), feats, strict=False):
        d = day.dropna(subset=[f])
        for _pup, g in d.groupby("animal_id"):
            g = g.sort_values("age")
            ax.plot(
                g["age"], g[f], color=LITTER_COLOURS[g["litter_id"].iloc[0]], alpha=0.25, lw=0.7
            )
        for lit, g in d.groupby("litter_id"):
            m = g.groupby("age")[f].median()
            ax.plot(m.index, m.to_numpy(), color=LITTER_COLOURS[lit], lw=2, label=lit)
        ax.set_title(label(f), fontsize=8)
        ax.set_xlabel("postnatal day", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Confirmed features by age (thin: individual pups; thick: litter median)", fontsize=10
    )
    _save(fig, out)


def identity_figure(run: Path, out: Path) -> None:
    t = pd.read_csv(run / "m9_identity_by_band.csv")
    bands = ["P1-7", "P8-14", "P15-21"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for lit, g in t.groupby("litter"):
        g = g.set_index("band").reindex(bands)
        x = np.arange(len(bands))
        ax.plot(
            x, g["balanced_accuracy"], "o-", color=LITTER_COLOURS[lit], label=f"{lit} (observed)"
        )
        ax.plot(
            x,
            g["null_mean"],
            "x--",
            color=LITTER_COLOURS[lit],
            alpha=0.6,
            label=f"{lit} (shuffled)",
        )
    ax.set_xticks(range(len(bands)), bands)
    ax.set_ylabel("which-pup accuracy on an unseen day")
    ax.set_title("Individual vocal signatures by age band", fontsize=10)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    _save(fig, out)


def example_calls(
    run: Path, images_run: Path, features_run: Path, feature: str, out: Path, rotation: str = "E2L2"
) -> None:
    """Real call spectrograms at the low and high end of one per-call score (held-out litter)."""
    from rat_usv.evaluation.m9 import beyond_contour, load_calls, project

    calls, curves = load_calls(features_run)
    idx = pd.read_parquet(images_run / "calls_index.parquet")
    imgs = np.load(images_run / "calls_images.npy", mmap_mode="r")
    pos = idx.reset_index().merge(calls.reset_index(names="k"), on=["wav_path", "call_index"])
    name = feature.replace("_med", "")
    if name.startswith(("f0m_", "envm_")):
        which = "f0" if name.startswith("f0m_") else "env"
        summary = json.loads((run / "m9_confirmation.json").read_text())
        names = summary["fpca"][rotation][which]["names"]
        with np.load(run / f"m9_bases_holdout_{rotation}.npz") as z:
            basis = {"mean": z[f"{which}_mean"], "modes": z[f"{which}_modes"]}
        x = curves["f0" if which == "f0" else "envelope"][calls["row"].to_numpy()].astype(float)
        score = project(x, basis)[:, names.index(name.split("_", 1)[1])]
    else:
        score = beyond_contour(calls, curves)[name.replace("share_", "has_")].to_numpy()
    pos["score"] = score[pos["k"].to_numpy()]
    pos = pos[pos["litter_id_x"] == rotation].dropna(subset=["score"])
    lo, hi = (
        pos.nsmallest(400, "score").sample(8, random_state=0),
        pos.nlargest(400, "score").sample(8, random_state=0),
    )
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for r, (title, sel) in enumerate((("low", lo), ("high", hi))):
        for c, (_, row) in enumerate(sel.iterrows()):
            ax = axes[r, c]
            ax.imshow(imgs[int(row["index"])], cmap="magma", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"P{int(row['postnatal_age_days_x'])}", fontsize=7)
        axes[r, 0].set_ylabel(f"{title} score", fontsize=8)
    fig.suptitle(
        f"Real calls (held-out {rotation}) at the extremes of {label(feature)}", fontsize=10
    )
    _save(fig, out)
