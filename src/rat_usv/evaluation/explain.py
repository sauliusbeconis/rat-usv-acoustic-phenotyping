"""M7 explanation chain: learned latent -> call families -> acoustic properties -> real calls.

Families are k-means clusters of VAE call embeddings fitted on the training litters of each
leave-one-litter-out fold and applied to the held-out litter. K is fixed a priori (8).
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rat_usv.evaluation.controls import add_derived, feature_sets
from rat_usv.evaluation.m4 import fit_predict, paired_bootstrap, per_pup_mae
from rat_usv.evaluation.splits import load_splits
from rat_usv.representations import vae

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K = 8
PROFILE = ("duration_ms", "f0_median_khz", "bandwidth_khz", "slope_khz_per_s", "harmonic_share",
           "n_jumps", "tonality", "peak_snr_db")  # fmt: skip


def fold_families(
    m4_run: Path, images: np.ndarray, call_idx: pd.DataFrame, litter: str, device: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings (all calls, this fold's VAE) and family labels (k-means on training litters)."""
    model = vae.VAE().to(device)
    model.load_state_dict(torch.load(m4_run / f"vae_holdout_{litter}.pt", map_location=device))
    z = vae.embed(model, images, device)
    train = (call_idx["litter_id"] != litter).to_numpy()
    scaler = StandardScaler().fit(z[train])
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(scaler.transform(z[train]))
    return z, km.predict(scaler.transform(z))


def run_explain(
    m4_run: Path, images_run: Path, controls_run: Path, features_run: Path, out: Path, seed: int = 0
) -> dict[str, Any]:
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    splits = load_splits(controls_run / "splits.json", digest)
    m3 = json.loads((controls_run / "controls_summary.json").read_text())
    birth = {k: pd.Timestamp(v).date() for k, v in m3["birth_dates"].items()}
    day = add_derived(pd.read_parquet(controls_run / "animal_day_with_background.parquet"), birth)
    classical = feature_sets(day, [])["classical"]
    call_idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    images = np.asarray(np.load(images_run / "calls_images.npy", mmap_mode="r"))
    feats = pd.read_parquet(features_run / "call_features.parquet")
    call_idx = call_idx.merge(
        feats[["wav_path", "call_index", *PROFILE]], on=["wav_path", "call_index"], how="left"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    held_family = np.full(len(call_idx), -1)
    held_dist = np.full(len(call_idx), np.nan)
    preds = {k: np.full(len(day), np.nan) for k in ("classical", "classical_families")}
    family_cols = [f"family_{i}" for i in range(K)]
    for fold in splits["leave_one_litter_out"]:
        litter = fold["litter"]
        z, fam = fold_families(m4_run, images, call_idx, litter, device, seed)
        held = (call_idx["litter_id"] == litter).to_numpy()
        held_family[held] = fam[held]
        zs = (z - z.mean(0)) / z.std(0)
        for f in range(K):
            m = held & (fam == f)
            if m.any():
                held_dist[m] = np.linalg.norm(zs[m] - zs[fam == f].mean(0), axis=1)
        props = (
            pd.DataFrame({"wav_path": call_idx["wav_path"], "family": fam})
            .pivot_table(index="wav_path", columns="family", aggfunc="size", fill_value=0)
            .reindex(columns=range(K), fill_value=0)
        )
        props = props.div(props.sum(axis=1), axis=0)
        props.columns = family_cols
        d = day.merge(props, left_on="wav_path", right_index=True, how="left")
        test = (d["litter_id"] == litter).to_numpy()
        preds["classical"][test] = fit_predict(d[~test], d[test], classical)
        preds["classical_families"][test] = fit_predict(d[~test], d[test], classical + family_cols)

    # Note: family indices are fold-specific; profiles below are per (fold, family) for held-out calls.
    call_idx["family"] = held_family
    call_idx["fold_litter"] = call_idx["litter_id"]
    call_idx["dist_to_centroid"] = held_dist
    scored = day[["animal_id", "age"]].copy()
    for k, v in preds.items():
        scored[f"pred_{k}"] = v
    mae = {k: per_pup_mae(scored, f"pred_{k}") for k in preds}
    m4 = pd.read_parquet(m4_run / "predictions_lolo.parquet")
    mae_vae = per_pup_mae(m4, "pred_combined")
    rng = np.random.default_rng(seed)

    profiles = (
        call_idx.groupby(["fold_litter", "family"])
        .agg(n=("family", "size"), age_mean=("postnatal_age_days", "mean"),
             **{f"median_{p}": (p, "median") for p in PROFILE})
        .reset_index()
    )  # fmt: skip
    profiles.to_csv(out / "family_profiles.csv", index=False)
    call_idx.to_parquet(out / "held_out_call_families.parquet")
    _montage(call_idx, images, out)
    _composition_figure(call_idx, out)
    return {
        "m4_run": m4_run.name,
        "k": K,
        "per_pup_mae_days": {k: round(float(v.mean()), 3) for k, v in mae.items()}
        | {"classical_plus_vae_m4": round(float(mae_vae.mean()), 3)},
        "classical_minus_classical_families": paired_bootstrap(mae["classical"], mae["classical_families"], rng),
        "classical_families_minus_classical_vae": paired_bootstrap(mae["classical_families"], mae_vae, rng),
        "share_of_vae_gain_explained_by_families": round(
            float((mae["classical"].mean() - mae["classical_families"].mean())
                  / (mae["classical"].mean() - mae_vae.mean())), 3),
        "status": "explanatory (not a confirmatory test); families are fold-specific k-means on training litters",
    }  # fmt: skip


def _montage(call_idx: pd.DataFrame, images: np.ndarray, out: Path, per: int = 8) -> None:
    for litter, g in call_idx[call_idx["family"] >= 0].groupby("fold_litter"):
        fig, axs = plt.subplots(K, per, figsize=(per * 1.3, K * 1.4))
        for f in range(K):
            sub = g[g["family"] == f].nsmallest(per, "dist_to_centroid")
            for j in range(per):
                ax = axs[f, j]
                ax.axis("off")
                if j < len(sub):
                    ax.imshow(images[sub.index[j]], cmap="magma", aspect="auto")
            age = g.loc[g["family"] == f, "postnatal_age_days"].mean()
            axs[f, 0].set_title(
                f"family {f} (mean P{age:.0f}, n={int((g['family'] == f).sum())})",
                fontsize=7,
                loc="left",
            )
        fig.suptitle(
            f"Typical held-out calls per VAE family (held-out litter {litter}); 18-100 kHz, sqrt time stretch",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out / f"family_montage_{litter}.png", dpi=110)
        plt.close(fig)


def _composition_figure(call_idx: pd.DataFrame, out: Path) -> None:
    for litter, g in call_idx[call_idx["family"] >= 0].groupby("fold_litter"):
        comp = g.groupby(["postnatal_age_days", "family"]).size().unstack(fill_value=0)
        comp = comp.div(comp.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.stackplot(
            comp.index, comp.T.values, labels=[f"family {f}" for f in comp.columns], alpha=0.85
        )
        ax.set_xlabel("Postnatal day")
        ax.set_ylabel("Share of calls")
        ax.set_title(
            f"Repertoire composition by age (held-out litter {litter}; VAE families)", fontsize=8
        )
        ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(out / f"family_composition_{litter}.png", dpi=150)
        plt.close(fig)
