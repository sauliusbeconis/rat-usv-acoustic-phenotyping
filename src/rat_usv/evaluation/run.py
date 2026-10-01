"""`usv controls`: freeze splits, compute background features and run the M3 control suite."""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rat_usv.config import SourceConfig
from rat_usv.data.linkage import final_pup_recordings, load_recordings
from rat_usv.evaluation.controls import run_controls
from rat_usv.evaluation.splits import load_splits, make_splits, write_splits
from rat_usv.provenance.io import write_json
from rat_usv.signal.background import background_features, band_names

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run_m3(
    cfg: SourceConfig,
    features_run: Path,
    detect_run: Path,
    out: Path,
    permutations: int,
    workers: int,
) -> dict[str, Any]:
    assert cfg.linkage_dir is not None
    day = pd.read_parquet(features_run / "animal_day.parquet")
    recs = final_pup_recordings(load_recordings(cfg.linkage_dir))
    birth = {r.litter_id: r.birth_date for r in recs if r.litter_id and r.birth_date}
    animals = dict(day[["animal_id", "litter_id"]].drop_duplicates().itertuples(index=False))
    digest = write_splits(make_splits(animals), out / "splits.json")
    splits = load_splits(out / "splits.json", digest)

    windows = pq.read_table(detect_run / "background_windows.parquet").to_pylist()
    bg = background_features(cfg.source_root, windows, workers)
    names = band_names()
    bg_df = pd.DataFrame.from_dict(bg, orient="index", columns=names)
    bg_df["bg_broadband_db"] = bg_df[names].mean(axis=1)
    day = day.merge(bg_df, left_on="wav_path", right_index=True, how="left")
    day.to_parquet(out / "animal_day_with_background.parquet")

    result = run_controls(day, birth, splits, names, permutations)
    result.update({"splits_sha256": digest, "features_run": str(features_run),
                   "detect_run": str(detect_run), "permutations": permutations})  # fmt: skip
    write_json(out / "controls_summary.json", result)
    _figure(result, out / "age_decoding_controls.png")
    return result


def _figure(result: dict[str, Any], path: Path) -> None:
    names = list(result["age_decoding_loao"])
    obs = [result["age_decoding_loao"][n]["mae_days"] for n in names]
    lo = [result["age_decoding_loao"][n]["mae_ci95"][0] for n in names]
    hi = [result["age_decoding_loao"][n]["mae_ci95"][1] for n in names]
    null = [result["age_decoding_loao"][n]["permutation_null_mae_mean"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x, obs, color="#4c72b0", label="held-out animals (LOAO)")
    ax.errorbar(x, obs, yerr=[np.subtract(obs, lo), np.subtract(hi, obs)], fmt="none", color="k")
    ax.scatter(x, null, color="#d62728", zorder=3, label="within-animal age permutation (mean)")
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylabel("Age error, days (per-animal MAE)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Age decoding: features vs negative controls (95% CI over animals)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
