"""M7 part 2: latent -> measurable properties, within-family age change, acoustic landscape.

For each leave-one-litter-out fold (one VAE run): a call-level latent age score is fitted on the
training litters (ridge on standardised embeddings) and applied to the held-out litter. A linear
map from classical call features to that score is also fitted on training litters; its residual
on held-out calls is the part of the latent age signal that classical features do not explain.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import ndimage, stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rat_usv.evaluation.explain import PROFILE
from rat_usv.evaluation.splits import load_splits
from rat_usv.representations import vae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

PROFILE_ALL = (*PROFILE, "sinuosity")


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def run_links(m4_run: Path, images_run: Path, controls_run: Path, features_run: Path, explain_run: Path,
              out: Path) -> dict[str, Any]:  # fmt: skip
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    splits = load_splits(controls_run / "splits.json", digest)
    idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    feats = pd.read_parquet(features_run / "call_features.parquet")
    idx = idx.merge(
        feats[["wav_path", "call_index", *PROFILE_ALL]], on=["wav_path", "call_index"], how="left"
    )
    fam = pd.read_parquet(explain_run / "held_out_call_families.parquet")[
        ["wav_path", "call_index", "family"]
    ]
    idx = idx.merge(fam, on=["wav_path", "call_index"], how="left")
    images = np.asarray(np.load(images_run / "calls_images.npy", mmap_mode="r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    age = idx["postnatal_age_days"].to_numpy(float)
    C = idx[list(PROFILE_ALL)].to_numpy(float)
    summary: dict[str, Any] = {"m4_run": m4_run.name, "folds": {}}
    heat = []
    for fold in splits["leave_one_litter_out"]:
        litter = fold["litter"]
        model = vae.VAE().to(device)
        model.load_state_dict(torch.load(m4_run / f"vae_holdout_{litter}.pt", map_location=device))
        z = vae.embed(model, images, device)
        train = (idx["litter_id"] != litter).to_numpy()
        held = ~train
        zs = StandardScaler().fit(z[train]).transform(z)
        score_model = Ridge(alpha=10.0).fit(zs[train], age[train])
        s = score_model.predict(zs)
        c_med = np.nanmedian(C[train], axis=0)
        Cf = np.where(np.isfinite(C), C, c_med)
        cs = StandardScaler().fit(Cf[train])
        lin = Ridge(alpha=1.0).fit(cs.transform(Cf[train]), s[train])
        s_classical = lin.predict(cs.transform(Cf))
        resid = s - s_classical
        fold_res = {
            "held_out_calls": int(held.sum()),
            "rho_latent_score_age": round(_rho(s[held], age[held]), 3),
            "rho_classical_part_age": round(_rho(s_classical[held], age[held]), 3),
            "rho_residual_age": round(_rho(resid[held], age[held]), 3),
            "r2_latent_score_from_classical": round(
                float(1 - np.var(resid[held]) / np.var(s[held])), 3
            ),
            "within_family_rho_residual_age": {
                int(f): round(
                    _rho(
                        resid[held & (idx["family"] == f).to_numpy()],
                        age[held & (idx["family"] == f).to_numpy()],
                    ),
                    3,
                )
                for f in sorted(idx.loc[held, "family"].dropna().unique())
            },  # fmt: skip
        }
        summary["folds"][litter] = fold_res
        heat.append(
            np.array(
                [
                    [_rho(z[held, d], C[held, j]) for j in range(C.shape[1])]
                    for d in range(z.shape[1])
                ]
            )
        )
        _residual_montage(idx, images, held, resid, age, litter, out)
        _landscape(idx, zs, s, score_model.coef_, held, train, litter, fold_res, out)
    _heatmap(np.mean(heat, axis=0), out)
    summary["mean_over_folds"] = {
        k: round(float(np.mean([f[k] for f in summary["folds"].values()])), 3)
        for k in ("rho_latent_score_age", "rho_classical_part_age", "rho_residual_age", "r2_latent_score_from_classical")
    }  # fmt: skip
    return summary


def _heatmap(h: np.ndarray, out: Path) -> None:
    order = np.argsort(-np.abs(h).max(axis=1))
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(h[order], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(
        range(len(PROFILE_ALL)),
        [p.replace("_", " ") for p in PROFILE_ALL],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(range(len(order)), [f"z{d:02d}" for d in order], fontsize=6)
    fig.colorbar(im, label="Spearman ρ (held-out calls, mean over folds)")
    ax.set_title("VAE latent dimensions vs classical call properties", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latent_property_heatmap.png", dpi=150)
    plt.close(fig)


def _residual_montage(idx: pd.DataFrame, images: np.ndarray, held: np.ndarray, resid: np.ndarray,
                      age: np.ndarray, litter: str, out: Path, per: int = 6) -> None:  # fmt: skip
    bands = [(2, 5), (10, 13), (17, 20)]
    fig, axs = plt.subplots(len(bands) * 2, per, figsize=(per * 1.3, len(bands) * 2 * 1.35))
    for b, (lo, hi) in enumerate(bands):
        m = np.flatnonzero(held & (age >= lo) & (age <= hi) & np.isfinite(resid))
        order = m[np.argsort(resid[m])]
        for row, pick, label in (
            (2 * b, order[:per], "latent reads YOUNGER"),
            (2 * b + 1, order[-per:], "latent reads OLDER"),
        ):
            for j in range(per):
                axs[row, j].axis("off")
                if j < len(pick):
                    axs[row, j].imshow(images[pick[j]], cmap="magma", aspect="auto")
            axs[row, 0].set_title(
                f"P{lo}-P{hi}: {label} than classical features", fontsize=7, loc="left"
            )
    fig.suptitle(
        f"Calls where the VAE age signal departs most from classical features (held-out {litter})",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / f"residual_montage_{litter}.png", dpi=110)
    plt.close(fig)


def _landscape(idx: pd.DataFrame, zs: np.ndarray, s: np.ndarray, w: np.ndarray, held: np.ndarray,
               train: np.ndarray, litter: str, res: dict[str, Any], out: Path) -> None:  # fmt: skip
    """Class 3 conceptual figure, built only from held-out data in a training-fitted frame."""
    u = w / np.linalg.norm(w)
    orth = zs[train] - np.outer(zs[train] @ u, u)
    _, _, vt = np.linalg.svd(orth - orth.mean(0), full_matrices=False)
    y = zs @ vt[0]
    x = s
    xh, yh = x[held], y[held]
    xr = np.percentile(xh, [1, 99])
    yr = np.percentile(yh, [1, 99])
    H, xe, ye = np.histogram2d(xh, yh, bins=80, range=[xr, yr], density=True)
    H = ndimage.gaussian_filter(H, 2.0)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    d = pd.DataFrame({"x": xh, "y": yh, "animal": idx.loc[held, "animal_id"].to_numpy(),
                      "age": idx.loc[held, "postnatal_age_days"].to_numpy()})  # fmt: skip
    g_ad = d.groupby(["animal", "age"])
    traj = g_ad[["x", "y"]].mean()[g_ad.size() >= 5].reset_index()  # pup-days with >= 5 calls
    pop = d.groupby("age")[["x", "y"]].mean()
    note = (f"Held-out litter {litter}. Conceptual view (class 3), built from held-out calls only. "
            f"x: latent age score (fit on other litters), ρ with age {res['rho_latent_score_age']}; "
            f"residual beyond classical features ρ {res['rho_residual_age']}. Evidence: decisions 0013/0015.")  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.contourf(xc, yc, H.T, levels=14, cmap="Greys")
    ax.contour(xc, yc, H.T, levels=14, colors="k", linewidths=0.3, alpha=0.5)
    for _, g in traj.groupby("animal"):  # individual pups, 3-day rolling mean (display only)
        g = g.sort_values("age")[["x", "y"]].rolling(3, center=True, min_periods=2).mean()
        ax.plot(g["x"], g["y"], color="tab:orange", alpha=0.3, lw=0.8)
    sc = ax.scatter(pop["x"], pop["y"], c=pop.index, cmap="coolwarm", s=40, zorder=3, edgecolor="k")
    ax.plot(pop["x"], pop["y"], color="k", lw=1.5, zorder=2)
    fig.colorbar(sc, label="Postnatal day (litter mean)")
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel("Latent age score (days)")
    ax.set_ylabel("Main latent variation orthogonal to age")
    ax.set_title(note, fontsize=6.5, wrap=True)
    fig.tight_layout()
    fig.savefig(out / f"landscape_2d_{litter}.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax3 = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    ax3.plot_surface(X, Y, H, cmap=cm.terrain, linewidth=0, antialiased=True, alpha=0.85)

    def height(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(xc, px), 0, len(xc) - 1)
        j = np.clip(np.searchsorted(yc, py), 0, len(yc) - 1)
        return H[i, j]

    top = np.full(len(pop), H.max() * 1.15)  # trajectory drawn above the terrain ...
    ax3.plot(pop["x"], pop["y"], top, color="k", lw=2.5)
    ax3.scatter(
        pop["x"], pop["y"], top, c=pop.index, cmap="coolwarm", s=36, depthshade=False, edgecolor="k"
    )
    ground = height(pop["x"].to_numpy(), pop["y"].to_numpy())  # ... and its trace on the surface
    ax3.plot(pop["x"], pop["y"], ground, color="k", lw=1, ls="--")
    for xi, yi, gi in zip(pop["x"], pop["y"], ground, strict=True):
        ax3.plot([xi, xi], [yi, yi], [gi, H.max() * 1.15], color="grey", lw=0.4)
    ax3.set_xlabel("latent age score")
    ax3.set_ylabel("orthogonal variation")
    ax3.set_zlabel("call density")
    ax3.view_init(elev=28, azim=-115)
    ax3.set_title(
        f"Acoustic landscape, development P1→P21 (held-out {litter}; conceptual, class 3)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out / f"landscape_3d_{litter}.png", dpi=170)
    plt.close(fig)


def write_summary(summary: dict[str, Any], out: Path) -> None:
    (out / "latent_links_summary.json").write_text(json.dumps(summary, indent=1))
