"""M4 representation ladder vs classical features under leave-one-litter-out.

Protocol v1 (locked): encoders are fitted on the training litters of each fold only; the
same fixed ridge decoder is used for every feature set; the primary statistic is the paired
per-pup MAE difference (classical - combined) with a pup-level bootstrap CI.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from rat_usv.evaluation.controls import add_derived, feature_sets, regressor
from rat_usv.evaluation.splits import load_splits
from rat_usv.provenance.io import write_json
from rat_usv.representations import encoders, vae


def pooled(index: pd.DataFrame, z: np.ndarray, key: str = "wav_path") -> pd.DataFrame:
    """Mean and SD of embeddings per recording (animal-day)."""
    cols = [f"z{i:02d}" for i in range(z.shape[1])]
    df = pd.DataFrame(z, columns=cols)
    df[key] = index[key].to_numpy()
    g = df.groupby(key)[cols]
    mean = g.mean().add_prefix("mean_")
    sd = g.std().fillna(0.0).add_prefix("sd_")
    return mean.join(sd)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    model = regressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train[cols].to_numpy(float),
            train["age"].to_numpy(float),
            groups=train["animal_id"].to_numpy(),
        )
    return model.predict(test[cols].to_numpy(float))


def per_pup_mae(d: pd.DataFrame, pred_col: str) -> pd.Series:
    return (d[pred_col] - d["age"]).abs().groupby(d["animal_id"]).mean()


def paired_bootstrap(
    a: pd.Series, b: pd.Series, rng: np.random.Generator, n: int = 5000
) -> dict[str, Any]:
    """Mean of (a - b) over pups with a pup-level bootstrap 95% CI (positive = b better)."""
    diff = (a - b).to_numpy()
    boots = [rng.choice(diff, diff.size, replace=True).mean() for _ in range(n)]
    return {
        "mean_mae_difference_days": round(float(diff.mean()), 3),
        "ci95": [
            round(float(np.percentile(boots, 2.5)), 3),
            round(float(np.percentile(boots, 97.5)), 3),
        ],
        "pups_improved": int((diff > 0).sum()),
        "n_pups": int(diff.size),
    }


def run_m4(
    kind: str, images_run: Path, controls_run: Path, out: Path, epochs: int, seed: int = 0
) -> dict[str, Any]:
    """kind: 'vae' (rung 1), 'cnn' (rung 2) or 'ssl' (rung 3)."""
    digest = (controls_run / "splits.sha256").read_text().split()[0]
    splits = load_splits(controls_run / "splits.json", digest)
    summary_m3 = json.loads((controls_run / "controls_summary.json").read_text())
    birth = {k: pd.Timestamp(v).date() for k, v in summary_m3["birth_dates"].items()}
    day = add_derived(pd.read_parquet(controls_run / "animal_day_with_background.parquet"), birth)
    classical = feature_sets(day, [])["classical"]

    call_idx = pq.read_table(images_run / "calls_index.parquet").to_pandas()
    call_img = np.load(images_run / "calls_images.npy", mmap_mode="r")
    bg_idx = pq.read_table(images_run / "background_index.parquet").to_pandas()
    bg_img = np.load(images_run / "background_images.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    preds = {
        k: np.full(len(day), np.nan)
        for k in (
            "classical",
            "learned",
            "combined",
            "background_learned",
            "clean_classical",
            "clean_combined",
        )
    }
    histories = {}
    for fold in splits["leave_one_litter_out"]:
        test_litter = fold["litter"]
        train_calls = np.flatnonzero((call_idx["litter_id"] != test_litter).to_numpy())
        all_calls = np.asarray(call_img)
        if kind == "vae":
            res = vae.train_vae(all_calls, train_calls, epochs, seed, device)
            histories[test_litter] = res.history
            torch.save(res.model.state_dict(), out / f"{kind}_holdout_{test_litter}.pt")
            z_calls = vae.embed(res.model, all_calls, device)
            z_bg = vae.embed(res.model, np.asarray(bg_img), device)
        else:
            ages = call_idx["postnatal_age_days"].to_numpy(np.float32)
            enc = encoders.train_encoder(kind, all_calls, train_calls, ages, epochs, seed, device)
            histories[test_litter] = enc.history
            torch.save(enc.encoder.state_dict(), out / f"{kind}_holdout_{test_litter}.pt")
            z_calls = encoders.embed_encoder(enc.encoder, all_calls, device)
            z_bg = encoders.embed_encoder(enc.encoder, np.asarray(bg_img), device)
        emb = pooled(call_idx, z_calls)
        bg = pooled(bg_idx, z_bg).add_prefix("bg_")
        d = day.merge(emb, left_on="wav_path", right_index=True, how="left")
        d = d.merge(bg, left_on="wav_path", right_index=True, how="left")
        learned_cols = list(emb.columns)
        bg_cols = [c for c in bg.columns if c.startswith("bg_mean_")]
        test = (d["litter_id"] == test_litter).to_numpy()
        tr, te = d[~test], d[test]
        preds["classical"][test] = fit_predict(tr, te, classical)
        preds["learned"][test] = fit_predict(tr, te, learned_cols)
        preds["combined"][test] = fit_predict(tr, te, classical + learned_cols)
        preds["background_learned"][test] = fit_predict(tr, te, bg_cols)
        # Sensitivity: decoders refitted and scored without edit-suspected recordings.
        keep = ~d["edit_suspected"].to_numpy()
        tr_c, te_c = d[~test & keep], d[test & keep]
        rows = np.flatnonzero(test & keep)
        preds["clean_classical"][rows] = fit_predict(tr_c, te_c, classical)
        preds["clean_combined"][rows] = fit_predict(tr_c, te_c, classical + learned_cols)

    scored = day[["animal_id", "litter_id", "age", "edit_suspected", "wav_path"]].copy()
    for k, v in preds.items():
        scored[f"pred_{k}"] = v
    scored.to_parquet(out / "predictions_lolo.parquet")
    mae = {k: per_pup_mae(scored.dropna(subset=[f"pred_{k}"]), f"pred_{k}") for k in preds}
    result_summary: dict[str, Any] = {
        "protocol_sha256": (Path("protocol/LOCK.sha256").read_text().split()[0]
                            if Path("protocol/LOCK.sha256").exists() else None),
        "splits_sha256": digest,
        "split": "leave_one_litter_out (scored per pup)",
        "rung": kind,
        "encoder": (vae.config() if kind == "vae" else encoders.config(kind))
        | {"epochs": epochs, "seed": seed, "device": device},
        "per_pup_mae_days": {k: round(float(v.mean()), 3) for k, v in mae.items()},
        "primary_classical_minus_combined": paired_bootstrap(mae["classical"], mae["combined"], rng),
        "classical_minus_learned": paired_bootstrap(mae["classical"], mae["learned"], rng),
        "background_learned_spearman": round(
            float(scored[["pred_background_learned", "age"]].corr("spearman").iat[0, 1]), 3
        ),
    }  # fmt: skip
    result_summary["no_edit_suspected_classical_minus_combined"] = paired_bootstrap(
        mae["clean_classical"], mae["clean_combined"], rng
    )
    ci = result_summary["primary_classical_minus_combined"]["ci95"]
    result_summary["h1_criteria"] = {
        "ci_excludes_zero_and_positive": ci[0] > 0,
        "background_does_not_decode_age_across_litters": abs(
            result_summary["background_learned_spearman"]
        )
        < 0.2,
        "holds_without_edit_suspected": result_summary[
            "no_edit_suspected_classical_minus_combined"
        ]["ci95"][0]
        > 0,
    }
    write_json(out / f"m4_{kind}_summary.json", result_summary)
    write_json(out / "training_history.json", histories)
    return result_summary
