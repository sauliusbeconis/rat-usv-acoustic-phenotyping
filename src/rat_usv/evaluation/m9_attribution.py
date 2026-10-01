"""M9 work package 5 (exploratory): where in the call image is the VAE's residual age signal?

Per leave-one-litter-out fold (VAE seed 0 of M4), on training-litter calls only:
1. a per-call baseline ridge predicts age from classical + H-M7a shape call features;
2. a residual ridge predicts (age - baseline) from the VAE embedding: the "residual score".
On held-out calls, image regions defined by the call's own f0 contour (fundamental or
harmonic band x start/middle/end third) are occluded (set to the noise floor, 0) and the
change in residual score is recorded. Latent traversals decode family centroids moved
along the residual direction. Requires the `ml` group (torch).
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rat_usv.representations import vae
from rat_usv.representations.images import MAX_DUR_S, SIZE
from rat_usv.signal.detect import DetectorConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CALL_FEATURES = ("log_duration", "f0_median_khz", "f0_min_khz", "f0_max_khz", "bandwidth_khz",
                 "slope_khz_per_s", "sinuosity", "tonality", "f0_start_khz", "f0_end_khz",
                 "onset_sweep_khz", "offset_sweep_khz", "u_index_khz", "curvature_khz",
                 "fm_residual_sd_khz", "fm_extrema_per_10ms")  # fmt: skip
REGIONS = (
    "fund_start",
    "fund_middle",
    "fund_end",
    "harm_start",
    "harm_middle",
    "harm_end",
    "outside",
)
BAND_KHZ = 3.0
N_OCCLUDE = 3000  # held-out calls per fold
AGE_BANDS = {"P1-7": (1, 7), "P8-14": (8, 14), "P15-21": (15, 21)}


def freq_grid(det: DetectorConfig, fs: int = 214285) -> np.ndarray:
    f = np.fft.rfftfreq(det.nperseg, 1 / fs)
    f = f[(f >= det.fmin_hz) & (f <= min(det.fmax_hz, fs / 2))] / 1000
    return np.linspace(float(f[0]), float(f[-1]), SIZE)


def region_masks(
    start: float, end: float, raw_t: np.ndarray, raw_f0: np.ndarray, grid_f: np.ndarray
) -> dict[str, np.ndarray]:
    """Boolean SIZE x SIZE masks (row 0 = highest frequency), following images.to_image."""
    dur = end - start
    centre = start + dur / 2
    span = max(float(np.sqrt(min(dur, MAX_DUR_S) * MAX_DUR_S)), min(dur, MAX_DUR_S))
    grid_t = np.linspace(centre - span / 2, centre + span / 2, SIZE)
    f0_t = np.interp(grid_t, raw_t, raw_f0, left=np.nan, right=np.nan)
    tau = (grid_t - raw_t[0]) / max(float(raw_t[-1] - raw_t[0]), 1e-9)
    third = np.select([tau < 1 / 3, tau < 2 / 3], [0, 1], 2)
    ff = grid_f[::-1][:, None]
    masks: dict[str, np.ndarray] = {}
    used = np.zeros((SIZE, SIZE), dtype=bool)
    for name, mult in (("fund", 1.0), ("harm", 2.0)):
        band = np.abs(ff - mult * f0_t[None, :]) <= BAND_KHZ
        for k, part in enumerate(("start", "middle", "end")):
            m = band & (third[None, :] == k) & np.isfinite(f0_t)[None, :]
            masks[f"{name}_{part}"] = m
            used |= m
    masks["outside"] = ~used
    return masks


def _ridge() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))
    )


@torch.no_grad()
def _scores(model: vae.VAE, imgs: np.ndarray, reg: Any, device: str) -> np.ndarray:
    z = vae.embed(model, imgs, device)
    return np.asarray(reg.predict(z))


def run_attribution(
    m4_run: Path,
    images_run: Path,
    features_run: Path,
    explain_run: Path,
    det: DetectorConfig,
    out: Path,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    images = np.load(images_run / "calls_images.npy", mmap_mode="r")
    feats = pd.read_parquet(features_run / "call_features.parquet")
    feats["log_duration"] = np.log(feats["duration_ms"])
    cidx = pd.read_parquet(features_run / "call_curves_index.parquet")
    with np.load(features_run / "call_curves.npz") as z:
        raw_t, raw_f0, off = z["raw_t"], z["raw_f0"], z["raw_t_offsets"]
    idx = idx.merge(
        feats[["wav_path", "call_index", *CALL_FEATURES]], on=["wav_path", "call_index"], how="left"
    )
    idx = idx.merge(
        cidx[["wav_path", "call_index", "row"]], on=["wav_path", "call_index"], how="left"
    )
    fam = pd.read_parquet(explain_run / "held_out_call_families.parquet")[
        ["wav_path", "call_index", "family"]
    ]
    idx = idx.merge(fam, on=["wav_path", "call_index"], how="left")
    age = idx["postnatal_age_days"].to_numpy(float)
    grid_f = freq_grid(det)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_imgs = np.asarray(images)
    rows, folds, traversals = [], {}, {}
    for litter in sorted(idx["litter_id"].unique()):
        train = (idx["litter_id"] != litter).to_numpy()
        model = vae.VAE().to(device)
        model.load_state_dict(torch.load(m4_run / f"vae_holdout_{litter}.pt", map_location=device))
        z = vae.embed(model, all_imgs, device)
        x = idx[list(CALL_FEATURES)].to_numpy(float)
        base = _ridge().fit(x[train], age[train])
        resid = age - base.predict(x)
        reg = _ridge().fit(z[train], resid[train])
        s = reg.predict(z)
        held = np.flatnonzero(~train)
        folds[litter] = {
            "baseline_r_held_out": round(float(stats.spearmanr(base.predict(x[held]), age[held]).statistic), 3),
            "residual_score_vs_residual_rho_held_out": round(float(stats.spearmanr(s[held], resid[held]).statistic), 3),
        }  # fmt: skip
        pick = rng.choice(held, min(N_OCCLUDE, held.size), replace=False)
        base_img = all_imgs[pick]
        occluded = {r: base_img.copy() for r in REGIONS}
        area = {r: np.zeros(pick.size) for r in REGIONS}
        for i, j in enumerate(pick):
            r0 = int(idx.at[j, "row"])
            t = raw_t[off[r0] : off[r0 + 1]].astype(float)
            f = raw_f0[off[r0] : off[r0 + 1]].astype(float)
            masks = region_masks(
                float(idx.at[j, "start_s"]), float(idx.at[j, "end_s"]), t, f, grid_f
            )
            for r in REGIONS:
                occluded[r][i][masks[r]] = 0
                area[r][i] = masks[r].mean()
        s0 = _scores(model, base_img, reg, device)
        for r in REGIONS:
            d = s0 - _scores(model, occluded[r], reg, device)
            for i, j in enumerate(pick):
                rows.append({"fold": litter, "call": int(j), "region": r, "delta": float(d[i]),
                             "area": float(area[r][i]), "age": age[j], "family": idx.at[j, "family"],
                             "resid": resid[j], "score": s0[i]})  # fmt: skip
        traversals[litter] = _traversal(
            model, z[train], idx.loc[train, "family"].to_numpy(), reg, device
        )
    table = pd.DataFrame(rows)
    table.to_parquet(out / "m9_occlusion.parquet")
    summary = _summarise(table)
    summary["folds"] = folds
    _figures(table, traversals, out)
    return summary


@torch.no_grad()
def _traversal(
    model: vae.VAE, z: np.ndarray, family: np.ndarray, reg: Any, device: str
) -> np.ndarray:
    """Decoded images: rows = families, columns = steps along the residual-age direction."""
    ridge = reg[-1]
    scale = reg[1].scale_
    w = ridge.coef_ / scale  # direction in raw latent units
    w = w / np.linalg.norm(w)
    proj = z @ w
    steps = np.linspace(-3, 3, 7) * proj.std()
    out = []
    for f in range(int(np.nanmax(family)) + 1):
        sel = family == f
        if sel.sum() == 0:
            continue
        c = z[sel].mean(0)
        zz = np.stack([c + a * w for a in steps]).astype(np.float32)
        img = model.decode(torch.from_numpy(zz).to(device)).clamp(0, 1).cpu().numpy()[:, 0]
        out.append(img)
    return np.array(out)


def _summarise(t: pd.DataFrame) -> dict[str, Any]:
    t = t.assign(abs_delta=t["delta"].abs())
    by_region = t.groupby("region").agg(
        mean_abs_delta=("abs_delta", "mean"), mean_area=("area", "mean")
    )
    by_region["abs_delta_per_area"] = by_region["mean_abs_delta"] / by_region["mean_area"].clip(
        lower=1e-4
    )
    # Information carried: drop in rank correlation of the residual score with the true residual.
    info = {}
    for r, g in t.groupby("region"):
        rho0 = stats.spearmanr(g["score"], g["resid"]).statistic
        rho1 = stats.spearmanr(g["score"] - g["delta"], g["resid"]).statistic
        info[r] = round(float(rho0 - rho1), 4)
    band = pd.cut(t["age"], [0, 7, 14, 21], labels=list(AGE_BANDS))
    by_band = t.assign(band=band).pivot_table(
        index="region", columns="band", values="abs_delta", aggfunc="mean", observed=True
    )
    by_family = t.pivot_table(
        index="region", columns=["fold", "family"], values="abs_delta", aggfunc="mean"
    )
    return {
        "by_region": by_region.round(4).to_dict(orient="index"),
        "rho_drop_when_occluded": info,
        "by_age_band_mean_abs_delta": by_band.round(4).to_dict(orient="index"),
        "n_family_cells": int(by_family.shape[1]),
        "status": "exploratory (not in the confirmation design)",
    }  # fmt: skip


def _figures(t: pd.DataFrame, traversals: dict[str, np.ndarray], out: Path) -> None:
    t = t.assign(
        abs_delta=t["delta"].abs(), band=pd.cut(t["age"], [0, 7, 14, 21], labels=list(AGE_BANDS))
    )
    grid = t.pivot_table(
        index="region", columns="band", values="abs_delta", aggfunc="mean", observed=True
    )
    grid = grid.reindex(list(REGIONS))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(grid.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), [r.replace("_", " ") for r in grid.index])
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if v < grid.to_numpy().max() * 0.6 else "black",
            fontsize=8,
        )
    fig.colorbar(im, label="mean |change| of residual age score (days)")
    ax.set_title("Occlusion: where the VAE's extra age signal sits")
    fig.tight_layout()
    fig.savefig(out / "m9_occlusion_regions.png", dpi=150)
    plt.close(fig)
    for litter, imgs in traversals.items():
        n_f = imgs.shape[0]
        show = [0, imgs.shape[1] // 2, imgs.shape[1] - 1]
        fig, axes = plt.subplots(n_f, 4, figsize=(4 * 1.3, n_f * 1.2), squeeze=False)
        for i in range(n_f):
            for j, k in enumerate(show):
                axes[i, j].imshow(imgs[i, k], cmap="magma", vmin=0, vmax=1)
            diff = imgs[i, -1] - imgs[i, 0]
            lim = max(float(np.abs(diff).max()), 1e-3)
            axes[i, 3].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[i, 0].set_ylabel(f"family {i}", fontsize=6)
        for j, t in enumerate(("younger (−3 SD)", "centre", "older (+3 SD)", "older − younger")):
            axes[0, j].set_title(t, fontsize=7)
        fig.suptitle(f"Decoded calls along the residual-age direction (fold {litter})", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"m9_traversal_{litter}.png", dpi=120)
        plt.close(fig)
