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


if __name__ == "__main__":
    app()
