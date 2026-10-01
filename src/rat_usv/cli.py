"""Only implemented commands are exposed here. No training interface exists."""

import json
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq
import typer

from rat_usv.config import load_config
from rat_usv.data.crosswalk import seed_crosswalk
from rat_usv.data.inventory import inventory as build_inventory
from rat_usv.data.linkage import check_sources, load_recordings, summarize
from rat_usv.evaluation.run import run_m3
from rat_usv.evidence.pilot import extract_pilot, validate_bundle_file
from rat_usv.evidence.reconcile import reconcile_metadata as reconcile_records
from rat_usv.evidence.review import review_packet as build_review_packet
from rat_usv.evidence.search import metadata_search as search_metadata
from rat_usv.provenance.io import new_run, record_run, write_json
from rat_usv.provenance.validation import (
    validate_fit_scope,
    validate_registry,
    validate_splits,
    verify_locked_split,
    verify_source_hashes,
)
from rat_usv.qc.audio import audit as audit_wav
from rat_usv.qc.full_decode import full_decode_pilot as run_full_decode
from rat_usv.representations.images import build_images
from rat_usv.signal.detect import load_detector_config
from rat_usv.signal.features import run_features
from rat_usv.signal.pipeline import run_detection
from rat_usv.signal.review import build_review, score_review
from rat_usv.statistics.sanity import avisoft_sanity
from rat_usv.statistics.trajectory import run_baseline
from rat_usv.visualization.atlas import dataset_atlas as build_atlas

app = typer.Typer(
    no_args_is_help=True, help="Read-only source inspection and scientific manifest validation."
)
ConfigArg = Annotated[
    Path,
    typer.Option(
        "--config",
        exists=True,
        dir_okay=False,
        help="Local YAML config; paths resolve relative to it.",
    ),
]
InputArg = Annotated[Path, typer.Option(exists=True, dir_okay=False)]


@app.command("reconcile-metadata")
def reconcile_metadata(config: ConfigArg, evidence: InputArg, search_entities: InputArg) -> None:
    """Compare historical date and input-path assertions using verified derivatives."""
    cfg = load_config(config)
    out = new_run(cfg, "reconcile-metadata")
    try:
        reconcile_records(cfg, evidence, search_entities, out)
    except Exception as error:
        write_json(out / "fatal_failure.json", {"error": f"{type(error).__name__}: {error}"})
        if not (out / "run.json").exists():
            record_run(out, "reconcile-metadata-failed", cfg, [evidence, search_entities])
        typer.echo(out)
        raise typer.Exit(1) from error
    typer.echo(out)


@app.command("metadata-search")
def metadata_search(config: ConfigArg, inventory: InputArg, policy: InputArg) -> None:
    """Search a bounded hashed metadata tree without executing historical code."""
    cfg = load_config(config)
    out = new_run(cfg, "metadata-search")
    try:
        report = search_metadata(cfg, inventory, policy, out)
    except Exception as error:
        write_json(out / "fatal_failure.json", {"error": f"{type(error).__name__}: {error}"})
        if not (out / "run.json").exists():
            record_run(out, "metadata-search-failed", cfg, [inventory, policy])
        typer.echo(out)
        raise typer.Exit(1) from error
    typer.echo(out)
    if not report["success"]:
        raise typer.Exit(1)


@app.command("review-packet")
def review_packet(config: ConfigArg, evidence: InputArg) -> None:
    """Expand an existing extraction ledger into a private, traceable review packet."""
    cfg = load_config(config)
    out = new_run(cfg, "review-packet")
    try:
        build_review_packet(cfg, evidence, out)
    except Exception as error:
        write_json(out / "fatal_failure.json", {"error": f"{type(error).__name__}: {error}"})
        if not (out / "run.json").exists():
            record_run(out, "review-packet-failed", cfg, [config, evidence])
        typer.echo(out)
        raise typer.Exit(1) from error
    typer.echo(out)


@app.command("full-decode-pilot")
def full_decode_pilot(config: ConfigArg, evidence: InputArg, policy: InputArg) -> None:
    """Fully decode the bounded, previously exposed PCM16 WAV evidence selection."""
    cfg = load_config(config)
    out = new_run(cfg, "full-decode-pilot")
    try:
        report = run_full_decode(cfg, evidence, policy, out)
    except Exception as error:
        write_json(out / "fatal_failure.json", {"error": f"{type(error).__name__}: {error}"})
        if not (out / "run.json").exists():
            record_run(out, "full-decode-pilot-failed", cfg, [evidence, policy])
        typer.echo(out)
        raise typer.Exit(1) from error
    typer.echo(out)
    if not report["success"]:
        raise typer.Exit(1)


@app.command("evidence-pilot")
def evidence_pilot(config: ConfigArg, selection: InputArg) -> None:
    """Extract explicitly selected evidence into a new private review bundle."""
    cfg = load_config(config)
    out = new_run(cfg, "evidence-pilot")
    try:
        report = extract_pilot(cfg, selection, out)
    except Exception as error:
        write_json(out / "fatal_failure.json", {"error": f"{type(error).__name__}: {error}"})
        if not (out / "run.json").exists():
            record_run(out, "evidence-pilot-failed", cfg, [selection])
        typer.echo(out)
        raise typer.Exit(1) from error
    typer.echo(out)
    if not report["success"]:
        raise typer.Exit(1)


@app.command("validate-evidence")
def validate_evidence(bundle: InputArg) -> None:
    """Validate typed evidence, provenance references and sibling manifest digest."""
    try:
        validated = validate_bundle_file(bundle)
    except (OSError, ValueError) as error:
        typer.echo(f"Invalid evidence bundle: {error}")
        raise typer.Exit(1) from error
    typer.echo(
        json.dumps(
            {"schema_valid": True, "schema_version": validated.schema_version, "m0_ready": False}
        )
    )


@app.command()
def inventory(config: ConfigArg) -> None:
    """Inventory paths and bounded hashes; classify candidate source roles."""
    cfg = load_config(config)
    out = new_run(cfg, "inventory")
    build_inventory(cfg, out)
    typer.echo(out)


@app.command()
def audit(config: ConfigArg, inventory: InputArg) -> None:
    """Audit WAV headers and sampled windows from a source inventory."""
    cfg = load_config(config)
    out = new_run(cfg, "audit")
    audit_wav(cfg, inventory, out)
    typer.echo(out)


@app.command()
def crosswalk(config: ConfigArg, inventory: InputArg) -> None:
    """Seed unresolved WAV filename aliases; never merge biological identities."""
    cfg = load_config(config)
    out = new_run(cfg, "crosswalk")
    seed_crosswalk(cfg, inventory, out)
    typer.echo(out)


@app.command("dataset-atlas")
def dataset_atlas(
    config: ConfigArg,
    inventory: InputArg,
    audit: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Create a preliminary technical JSON/Markdown atlas with explicit unknowns."""
    cfg = load_config(config)
    out = new_run(cfg, "dataset-atlas")
    build_atlas(cfg, inventory, audit, out)
    typer.echo(out)


@app.command()
def validate(
    config: ConfigArg,
    registry: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    source_inventory: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    verify_hashes: Annotated[
        bool,
        typer.Option(help="Re-read every hashed source in the supplied baseline (explicit cost)."),
    ] = False,
    split: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    expected_split_sha256: Annotated[str | None, typer.Option()] = None,
    fit_log: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    litter_holdout: bool = False,
) -> None:
    """Validate registries, optional source hashes, locked splits and fit provenance."""
    if verify_hashes and source_inventory is None:
        raise typer.BadParameter("--verify-hashes requires --source-inventory")
    if (fit_log or expected_split_sha256 or litter_holdout) and split is None:
        raise typer.BadParameter("Split options require --split")
    cfg = load_config(config)
    out = new_run(cfg, "validate")
    report = validate_registry(registry)
    inputs = sorted(registry.glob("*.parquet"))
    if verify_hashes and source_inventory:
        report["source_hashes"] = verify_source_hashes(
            cfg.source_root, pq.read_table(source_inventory).to_pylist()
        )
        report["errors"].extend(report["source_hashes"]["failures"])
        inputs.append(source_inventory)
    if split:
        inputs.append(split)
        try:
            rows = json.loads(split.read_text())
            validate_splits(rows, litter_holdout)
            if expected_split_sha256:
                verify_locked_split(split, expected_split_sha256)
            else:
                report["warnings"].append("Split has no independent expected lock digest")
            if fit_log:
                inputs.append(fit_log)
                validate_fit_scope(json.loads(fit_log.read_text()), rows)
            else:
                report["warnings"].append("Fit provenance was not supplied")
        except (ValueError, KeyError, TypeError) as e:
            report["errors"].append(str(e))
    report["valid"] = not report["errors"]
    write_json(out / "validation.json", report)
    record_run(out, "validate", cfg, inputs)
    typer.echo(out)
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command("linkage-check")
def linkage_check(config: ConfigArg) -> None:
    """Validate the frozen recording linkage and confirm every WAV is present (stat only)."""
    cfg = load_config(config)
    if cfg.linkage_dir is None:
        raise typer.BadParameter("Config needs linkage_dir (folder containing recordings.csv)")
    rows = load_recordings(cfg.linkage_dir)
    report = summarize(rows)
    report["source_problems"] = check_sources(rows, cfg.source_root)
    out = new_run(cfg, "linkage-check")
    write_json(out / "linkage_check.json", report)
    record_run(out, "linkage-check", cfg, [cfg.linkage_dir / "recordings.csv"])
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "source_problems"}, indent=1))
    typer.echo(f"source problems: {len(report['source_problems'])}")
    typer.echo(out)
    if report["source_problems"]:
        raise typer.Exit(code=1)


@app.command()
def detect(
    config: ConfigArg,
    detector: InputArg = Path("configs/detection/default.yaml"),
    workers: Annotated[int, typer.Option(min=1)] = 8,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N recordings")] = None,
) -> None:
    """M1: detect calls in all final pup recordings (read-only) and compare with 2020 Avisoft."""
    cfg = load_config(config)
    det = load_detector_config(detector)
    out = new_run(cfg, "detect")
    summary = run_detection(cfg, det, out, workers, limit)
    record_run(out, "detect", cfg, [detector])
    typer.echo(json.dumps({k: summary[k] for k in ("recordings", "calls")}, indent=1))
    typer.echo(out)


RunDir = Annotated[Path, typer.Option(exists=True, file_okay=False)]


@app.command("detect-review")
def detect_review(config: ConfigArg, detect_run: RunDir) -> None:
    """Build a blind hand-check page (review.html) from a detect run's sampled events."""
    cfg = load_config(config)
    out = new_run(cfg, "detect-review")
    n = build_review(cfg, detect_run, out)
    record_run(out, "detect-review", cfg, [detect_run / "detection_summary.json"])
    typer.echo(f"{n} items -> {out / 'review.html'}")


@app.command("detect-score")
def detect_score(config: ConfigArg, review_run: RunDir, labels: InputArg) -> None:
    """Score the owner's labels.json against the review items (share of real calls per kind)."""
    cfg = load_config(config)
    items = json.loads((review_run / "review_items.json").read_text())
    out = new_run(cfg, "detect-score")
    result = score_review(items, json.loads(labels.read_text()))
    write_json(out / "detection_check.json", result)
    record_run(out, "detect-score", cfg, [review_run / "review_items.json", labels])
    typer.echo(json.dumps(result, indent=1))
    typer.echo(out)


@app.command()
def features(
    config: ConfigArg,
    detect_run: RunDir,
    detector: InputArg = Path("configs/detection/default.yaml"),
    workers: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
    """M2: per-call fundamental-contour features and per animal-day summaries."""
    cfg = load_config(config)
    out = new_run(cfg, "features")
    summary = run_features(cfg, load_detector_config(detector), detect_run, out, workers)
    record_run(out, "features", cfg, [detector, detect_run / "calls.parquet"])
    typer.echo(json.dumps(summary, indent=1))
    typer.echo(out)


@app.command()
def baseline(config: ConfigArg, features_run: RunDir) -> None:
    """M2: mixed-model age trajectories of classical animal-day features, with figures."""
    cfg = load_config(config)
    out = new_run(cfg, "baseline")
    summary = run_baseline(features_run, out)
    sanity = avisoft_sanity(cfg, features_run)
    write_json(out / "avisoft_sanity.json", sanity)
    record_run(out, "baseline", cfg, [features_run / "animal_day.parquet"])
    typer.echo(json.dumps(sanity, indent=1))
    for key, r in summary["results"].items():
        if "error" in r:
            typer.echo(f"{key}: FAILED {r['error']}")
        else:
            typer.echo(
                f"{key}: n={r['n_animals']}/{r['n_animal_days']} age p={r['age_lrt_p']:.2g} "
                f"ICC={r['animal_icc']:.2f}"
            )
    typer.echo(out)


@app.command()
def controls(
    config: ConfigArg,
    features_run: RunDir,
    detect_run: RunDir,
    permutations: Annotated[int, typer.Option(min=0)] = 100,
    workers: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
    """M3: freeze animal/litter splits and run negative controls on animal-day features."""
    cfg = load_config(config)
    out = new_run(cfg, "controls")
    result = run_m3(cfg, features_run, detect_run, out, permutations, workers)
    record_run(out, "controls", cfg, [features_run / "animal_day.parquet", out / "splits.json"])
    for name, r in result["age_decoding_loao"].items():
        typer.echo(
            f"LOAO {name}: MAE {r['mae_days']} {r['mae_ci95']} null {r['permutation_null_mae_mean']} p={r['permutation_p']:.3f}"
        )
    for name, r in result["age_decoding_lolo"].items():
        typer.echo(f"LOLO {name}: MAE {r['mae_days']} bias {r['bias_by_held_out_litter']}")
    typer.echo(
        json.dumps(
            {"date_vs_age": result["date_vs_age"], "identity": result["identity_decoding"]},
            indent=1,
        )
    )
    typer.echo(f"splits sha256 {result['splits_sha256']}")
    typer.echo(out)


@app.command("call-images")
def call_images(
    config: ConfigArg,
    detect_run: RunDir,
    features_run: RunDir,
    detector: InputArg = Path("configs/detection/default.yaml"),
    workers: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
    """M4 input: 128x128 whitened spectrogram images of confident calls and background crops."""
    cfg = load_config(config)
    out = new_run(cfg, "call-images")
    summary = build_images(
        cfg.source_root, detect_run, features_run, load_detector_config(detector), out, workers
    )
    write_json(out / "images_summary.json", summary)
    record_run(out, "call-images", cfg, [detector, features_run / "call_features.parquet"])
    typer.echo(json.dumps(summary, indent=1))
    typer.echo(out)


@app.command()
def m4(
    config: ConfigArg,
    kind: Annotated[str, typer.Option(help="vae | cnn | ssl")],
    images_run: RunDir,
    controls_run: RunDir,
    epochs: Annotated[int, typer.Option(min=1)] = 40,
    seed: int = 0,
) -> None:
    """M4 ladder rung (needs `--group ml`): learned embeddings vs classical, leave-one-litter-out."""
    if kind not in {"vae", "cnn", "ssl"}:
        raise typer.BadParameter("kind must be vae, cnn or ssl")
    from rat_usv.evaluation.m4 import run_m4  # torch is optional; import only here

    cfg = load_config(config)
    out = new_run(cfg, f"m4-{kind}")
    summary = run_m4(kind, images_run, controls_run, out, epochs, seed)
    inputs = [controls_run / "splits.json", images_run / "calls_index.parquet"]
    record_run(out, f"m4-{kind}", cfg, inputs)
    typer.echo(json.dumps(summary, indent=1))
    typer.echo(out)


@app.command()
def h2(
    config: ConfigArg,
    m4_table: InputArg,
    images_run: RunDir,
    controls_run: RunDir,
) -> None:
    """H2/H2b (needs `--group ml`): held-out trajectory structure and acoustic age vs weight."""
    from rat_usv.evaluation.h2 import json_safe, run_h2, run_h2b  # torch optional; import here

    cfg = load_config(config)
    assert cfg.linkage_dir is not None
    runs = [cfg.output_root / r["run"] for r in json.loads(m4_table.read_text())]
    out = new_run(cfg, "h2")
    result = json_safe(
        {
            "h2": run_h2(runs, images_run, controls_run),
            "h2b": run_h2b(runs, controls_run, cfg.linkage_dir),
        }
    )
    write_json(out / "h2_summary.json", result)
    record_run(out, "h2", cfg, [m4_table, controls_run / "splits.json"])
    typer.echo(json.dumps(result["h2"]["by_space"], indent=1))
    typer.echo(json.dumps(result["h2b"], indent=1))
    typer.echo(out)


@app.command()
def explain(
    config: ConfigArg,
    m4_run: RunDir,
    images_run: RunDir,
    controls_run: RunDir,
    features_run: RunDir,
) -> None:
    """M7 (needs `--group ml`): VAE call families, their acoustics, typical calls, explanatory test."""
    from rat_usv.evaluation.explain import run_explain  # torch optional; import here

    cfg = load_config(config)
    out = new_run(cfg, "explain")
    result = run_explain(m4_run, images_run, controls_run, features_run, out)
    write_json(out / "explain_summary.json", result)
    record_run(out, "explain", cfg, [controls_run / "splits.json"])
    typer.echo(json.dumps(result, indent=1))
    typer.echo(out)


@app.command()
def explorer(
    config: ConfigArg,
    m4_run: RunDir,
    images_run: RunDir,
    explain_run: RunDir,
    litter: str = "E2L3",
    points: Annotated[int, typer.Option(min=10)] = 300,
) -> None:
    """M7 exploratory explorer (needs `--group ml`): clickable latent map with spectrogram and audio."""
    from rat_usv.visualization.explorer import build_explorer  # torch optional; import here

    cfg = load_config(config)
    out = new_run(cfg, "explorer")
    result = build_explorer(cfg.source_root, m4_run, images_run, explain_run, litter, points, out)
    write_json(out / "explorer_summary.json", result)
    record_run(out, "explorer", cfg, [explain_run / "explain_summary.json"])
    typer.echo(f"{result['points']} points -> {out / 'explorer.html'}")


if __name__ == "__main__":
    app()
