"""Explicit selections, immutable per-run evidence bundles and full pilot-input hashes."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from rat_usv.config import HashPolicy, SourceConfig
from rat_usv.evidence import parsers
from rat_usv.provenance.io import Hasher, record_run, safe_source, sha256_file, write_json
from rat_usv.schemas.evidence import (
    PARSER_VERSION,
    Artifact,
    Assertion,
    EvidenceBundle,
    Observation,
    Relationship,
    canonical_json,
    stable_id,
)


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    kind: Literal["workbook", "wav", "script", "pdf", "archive"]
    lineage: str = "independence_unreviewed"
    pages: list[int] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    selection_reason: str


class StatementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    locator: str
    contains: str = Field(min_length=1)
    subject: str
    field: str
    value: JsonValue
    original_units: str | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)


class LinkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    recording: str
    export: str
    reason: str


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[2] = 2
    protocol: str
    baseline_inventory: str | None = None
    sources: list[SourceSpec] = Field(min_length=1, max_length=64)
    statements: list[StatementSpec] = Field(default_factory=list)
    candidate_links: list[LinkSpec] = Field(default_factory=list)
    unresolved_decisions: list[str] = Field(default_factory=list)
    max_source_bytes: int = Field(default=1024**3, gt=0, le=2 * 1024**3)


def signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def timestamp_review(rows: list[Observation]) -> dict[str, JsonValue]:
    """Retain alternative numeric conventions; flag disagreement without choosing one."""
    observations = [o for o in rows if o.kind == "timestamp_text"]
    dates = [o for o in observations if o.raw.get("field") == "origination_date"]
    times = [o for o in observations if o.raw.get("field") == "origination_time"]
    report: dict[str, JsonValue] = {
        "status": "UNRESOLVED",
        "observation_ids": [o.observation_id for o in observations],
        "timezone": None,
        "clock_accuracy": None,
        "comparisons": [],
    }
    if len(dates) != 1 or len(times) != 1:
        report["reason"] = "Absent or multiple bext date/time fields; no pairing inferred"
        return report
    bext = str(dates[0].raw["value"]).replace(":", "-") + " " + str(times[0].raw["value"])
    try:
        reference = datetime.strptime(bext, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        report["reason"] = "bext date/time is not interpretable under the documented convention"
        return report
    comparisons: list[JsonValue] = []
    for obs in observations:
        if obs.raw.get("field") != "TIME":
            continue
        interpretations: list[JsonValue] = []
        matches: list[JsonValue] = []
        for convention in ("%m/%d/%y %H:%M:%S", "%d/%m/%y %H:%M:%S"):
            try:
                candidate = datetime.strptime(str(obs.raw["value"]), convention)
            except ValueError:
                continue
            interpretations.append(
                {
                    "hypothesis": convention,
                    "candidate_timestamp": candidate.isoformat(),
                    "century_rule": "Python strptime two-digit-year rule; unconfirmed",
                }
            )
            if candidate == reference:
                matches.append(convention)
        comparisons.append(
            {
                "time_observation_id": obs.observation_id,
                "bext_observation_ids": [dates[0].observation_id, times[0].observation_id],
                "interpretations": interpretations,
                "matching_hypotheses": matches,
            }
        )
        if interpretations and not matches:
            report["status"] = "CONFLICT"
    report["comparisons"] = comparisons
    report["reason"] = (
        "Numeric date/time hypotheses only; agreement does not validate date convention or recorder clock"
    )
    return report


def script_assertions(artifact: Artifact, observations: list[Observation]) -> list[Assertion]:
    result = []
    binding = "unknown"
    recode_block: str | None = None
    in_code = not (artifact.member_name or artifact.source_path).lower().endswith(".rmd")
    for obs in observations:
        if obs.kind != "script_line":
            continue
        text = str(obs.raw["text"])
        if text.lstrip().startswith("```"):
            in_code = text.lstrip().startswith("```{r")
            continue
        if not in_code or text.lstrip().startswith("#"):
            continue
        assignment = re.match(r"\s*([\w.]+)\s*<-", text)
        if assignment:
            binding = assignment[1]
        if "fct_recode(" in text:
            recode_block = obs.locator
        if recode_block:
            day = re.match(r'\s*"(\d+)"\s*=\s*"(\d{2}\.\d{2})"', text)
            if day:
                payload = {
                    "subject": f"{artifact.lineage}:{binding}:{recode_block}:{day[2]}",
                    "field": "historical_day_label",
                    "value": day[1],
                    "observation_ids": [obs.observation_id],
                    "context": {
                        "binding": binding,
                        "recode_block": recode_block,
                        "date_token": day[2],
                        "year": None,
                        "postnatal_convention": None,
                    },
                }
                result.append(
                    Assertion.model_validate(
                        {"assertion_id": stable_id("assert", payload), **payload}
                    )
                )
            if re.match(r"\s*\)", text):
                recode_block = None
        sex = re.search(
            r'(pateles|patinai)_\d+_vada\s*<-.*vados_nr\s*==\s*("?)(\d+)\2\s*&.*individo_id\s*%in%\s*c\(([^)]*)\)',
            text,
        )
        if sex:
            for token in sex[4].split(","):
                original = token.strip()
                quoted = len(original) >= 2 and original[0] == original[-1] == '"'
                alias = original[1:-1] if quoted else original
                payload = {
                    "subject": f"{artifact.lineage}:vados_nr={sex[3]}:alias={alias}",
                    "field": "sex_recode",
                    "value": "female" if sex[1] == "pateles" else "male",
                    "observation_ids": [obs.observation_id],
                    "context": {
                        "alias": alias,
                        "group_token": sex[3],
                        "original_token": original,
                        "quoted_string": quoted,
                        "leading_zero_risk": not quoted,
                        "scope": "historical script group; not a verified animal",
                    },
                }
                result.append(
                    Assertion.model_validate(
                        {"assertion_id": stable_id("assert", payload), **payload}
                    )
                )
    return result


def interval_comparison(
    wav_rows: list[Observation], cells: list[Observation]
) -> dict[str, JsonValue]:
    """Compare complete sequences under explicit unit hypotheses; never certify a link."""
    labels = [
        o for o in wav_rows if o.kind == "label_line" and o.raw.get("candidate_start") is not None
    ]
    by_sheet: dict[str, list[Observation]] = defaultdict(list)
    for cell in cells:
        if cell.kind == "cell":
            by_sheet[cell.locator.rsplit("!", 1)[0]].append(cell)
    comparisons: list[JsonValue] = []
    for sheet, values in by_sheet.items():
        headers = {
            str(o.raw.get("value")): re.sub(r"\d+$", "", o.locator.rsplit("!", 1)[1])
            for o in values
            if o.raw.get("row_role") == "header"
        }
        starts, ends = headers.get("start time"), headers.get("end time")
        if not starts or not ends:
            continue
        indexed = {o.locator.rsplit("!", 1)[1]: o for o in values}
        event_rows = sorted(
            {
                int(o.locator.rsplit("!", 1)[1].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                for o in values
                if o.raw.get("row_role") == "numbered_event_candidate"
            }
        )
        deltas: list[JsonValue] = []
        for i, row in enumerate(event_rows):
            start, end = indexed[f"{starts}{row}"], indexed[f"{ends}{row}"]
            item: dict[str, JsonValue] = {
                "event_row": row,
                "start_observation_id": start.observation_id,
                "end_observation_id": end.observation_id,
                "label_observation_id": None,
                "start_delta_ms": None,
                "end_delta_ms": None,
            }
            if i < len(labels):
                label = labels[i]
                item["label_observation_id"] = label.observation_id
                try:
                    start_delta = float(str(start.raw["value"])) - 1000 * float(
                        str(label.raw["candidate_start"])
                    )
                    end_delta = float(str(end.raw["value"])) - 1000 * float(
                        str(label.raw["candidate_end"])
                    )
                    if not math.isfinite(start_delta) or not math.isfinite(end_delta):
                        raise ValueError("Nonfinite interval value")
                    item["start_delta_ms"], item["end_delta_ms"] = start_delta, end_delta
                except (TypeError, ValueError):
                    item["parse_error"] = "Non-numeric interval value; retained without filling"
            deltas.append(item)
        comparisons.append(
            {
                "sheet": sheet,
                "event_rows": len(event_rows),
                "label_intervals": len(labels),
                "same_count": len(event_rows) == len(labels),
                "pairs": deltas,
                "unpaired_label_ids": [o.observation_id for o in labels[len(event_rows) :]],
            }
        )
    return {
        "assumption": "Workbook units ms and LBL units s; unconfirmed, ordinal comparison only",
        "automatic_link_confirmation": False,
        "sheets": comparisons,
    }


def extract_pilot(config: SourceConfig, selection_path: Path, out: Path) -> dict[str, Any]:
    """Run only explicit inputs, with independent second hashes. Persist failures."""
    selection_path = selection_path.resolve(strict=True)
    selection = Selection.model_validate(yaml.safe_load(selection_path.read_text()))
    if len({s.path for s in selection.sources}) != len(selection.sources):
        raise ValueError("Duplicate source selection")
    if not out.resolve().is_relative_to(config.output_root):
        raise ValueError("Pilot output must be under configured output_root")
    if any(out.iterdir()):
        raise ValueError("Pilot output must be an empty new run")
    protocol = (selection_path.parent / selection.protocol).resolve(strict=True)
    paths = {s.path: safe_source(config.source_root, s.path) for s in selection.sources}
    for path in paths.values():
        rel = path.relative_to(config.source_root)
        if any(rel.is_relative_to(x) for x in config.exclude_relative):
            raise ValueError("Selected source is inside an excluded/output subtree")
    if sum(p.stat().st_size for p in paths.values()) > selection.max_source_bytes:
        raise ValueError("Pilot source hash/read budget exceeded")
    if any(p.stat().st_size > 512 * 1024**2 for p in paths.values()):
        raise ValueError("Pilot per-file hash/read budget exceeded")
    inputs = [selection_path, protocol]
    baseline = {}
    if selection.baseline_inventory:
        baseline_path = (selection_path.parent / selection.baseline_inventory).resolve(strict=True)
        inputs.append(baseline_path)
        baseline = {r["relative_path"]: r for r in pq.read_table(baseline_path).to_pylist()}
    write_json(out / "selection.json", selection.model_dump(mode="json"))
    (out / "protocol_snapshot.md").write_bytes(protocol.read_bytes())
    failures: list[dict[str, str]] = []
    manifest: list[dict[str, Any]] = []
    first = Hasher(HashPolicy(mode="full"))
    for name, path in paths.items():
        row: dict[str, Any] = {"source_path": name, "before": signature(path)}
        try:
            digest, status = first.compute(path)
            row.update(sha256=digest, hash_status=status)
            if status != "hashed":
                raise ValueError(status)
            previous = baseline.get(name)
            row["baseline_hash_comparison"] = "not_previously_hashed"
            if selection.baseline_inventory and previous is None:
                raise ValueError("Selected input absent from baseline inventory")
            if previous:
                if (
                    previous["size_bytes"] != row["before"]["size_bytes"]
                    or previous["mtime_ns"] != row["before"]["mtime_ns"]
                ):
                    raise ValueError("Selected input metadata disagrees with baseline")
                if previous.get("sha256"):
                    row["baseline_hash_comparison"] = (
                        "match" if digest == previous["sha256"] else "mismatch"
                    )
                    if digest != previous["sha256"]:
                        raise ValueError("Selected input hash disagrees with baseline")
        except (OSError, ValueError) as error:
            row["error"] = str(error)
            failures.append({"source": name, "stage": "initial_hash", "error": str(error)})
        manifest.append(row)
    write_json(out / "source_manifest.json", manifest)
    manifest_hash = sha256_file(out / "source_manifest.json")
    bundle = EvidenceBundle(manifest_sha256=manifest_hash, protocol_sha256=sha256_file(protocol))
    by_source: dict[str, list[Observation]] = defaultdict(list)

    def artifact(
        spec: SourceSpec,
        digest: str,
        size: int,
        member: str | None = None,
        ordinal: int | None = None,
        container_hash: str | None = None,
    ) -> Artifact:
        item = Artifact(
            artifact_id=stable_id("artifact", [spec.path, member, ordinal, digest]),
            source_path=spec.path,
            sha256=digest,
            size_bytes=size,
            member_name=member,
            member_index=ordinal,
            container_sha256=container_hash,
            lineage=spec.lineage,
        )
        bundle.artifacts.append(item)
        return item

    def observe(art: Artifact, values: Iterable[parsers.Raw]) -> list[Observation]:
        result = []
        for locator, kind, raw in values:
            obs = Observation(
                observation_id=stable_id(
                    "obs", [art.artifact_id, locator, kind, raw, PARSER_VERSION]
                ),
                artifact_id=art.artifact_id,
                locator=locator,
                kind=kind,
                raw=raw,
                manifest_sha256=manifest_hash,
            )
            bundle.observations.append(obs)
            by_source[art.source_path].append(obs)
            result.append(obs)
        return result

    for spec, entry in zip(selection.sources, manifest, strict=True):
        if entry.get("error"):
            continue
        path = paths[spec.path]
        art = artifact(spec, entry["sha256"], entry["before"]["size_bytes"])
        observe(
            art,
            [
                (
                    "file",
                    "artifact",
                    {"source_path": spec.path, "selection_reason": spec.selection_reason},
                )
            ],
        )
        try:
            if spec.kind == "wav":
                observe(
                    art,
                    [
                        (
                            "filename",
                            "recording_alias",
                            {
                                "alias": path.stem,
                                "namespace": str(Path(spec.path).parent),
                                "canonical_animal_id": None,
                            },
                        )
                    ],
                )
                with path.open("rb") as stream:
                    observe(art, parsers.wav(stream, path.stat().st_size))
                for obs in by_source[spec.path]:
                    if obs.kind == "timestamp_text":
                        timestamp = {
                            "subject": spec.path,
                            "field": "acquisition_timestamp_text",
                            "value": obs.raw["value"],
                            "context": {
                                "metadata_field": obs.raw["field"],
                                "locator": obs.locator,
                                "timezone": None,
                                "clock_accuracy": None,
                            },
                            "observation_ids": [obs.observation_id],
                        }
                        bundle.assertions.append(
                            Assertion.model_validate(
                                {"assertion_id": stable_id("assert", timestamp), **timestamp}
                            )
                        )
            elif spec.kind in {"script", "workbook"}:
                if path.stat().st_size > parsers.MAX_MEMBER:
                    raise ValueError("Text/workbook read budget exceeded")
                data = path.read_bytes()
                parsed = observe(
                    art,
                    parsers.script(data)
                    if spec.kind == "script"
                    else parsers.workbook(data, path.name),
                )
                if spec.kind == "script":
                    bundle.assertions.extend(script_assertions(art, parsed))
            elif spec.kind == "pdf":
                if not spec.pages or len(spec.pages) > 32 or any(p < 1 for p in spec.pages):
                    raise ValueError("PDF selection needs 1 to 32 positive page numbers")
                observe(art, parsers.pdf_pages(path, spec.pages))
            elif spec.kind == "archive":
                with parsers.BoundedZip(path) as package:
                    observe(art, parsers.archive_directory(package))
                    for name in spec.members:
                        matches = [
                            (i, info)
                            for i, info in enumerate(package.infos)
                            if info.filename == name
                        ]
                        if not matches:
                            failures.append(
                                {
                                    "source": spec.path,
                                    "stage": "archive_member",
                                    "error": f"Missing requested member: {name}",
                                }
                            )
                        for i, info in matches:
                            try:
                                data = package.read(i)
                                member = artifact(
                                    spec,
                                    hashlib.sha256(data).hexdigest(),
                                    len(data),
                                    name,
                                    i,
                                    art.sha256,
                                )
                                observe(
                                    member,
                                    [
                                        (
                                            "file",
                                            "artifact",
                                            {
                                                "member_name": name,
                                                "member_index": i,
                                                "header_offset": info.header_offset,
                                            },
                                        )
                                    ],
                                )
                                if name.endswith(".xlsx"):
                                    observe(member, parsers.workbook(data, name))
                                elif name.lower().endswith((".r", ".rmd")):
                                    parsed = observe(member, parsers.script(data))
                                    bundle.assertions.extend(script_assertions(member, parsed))
                            except Exception as error:
                                failures.append(
                                    {
                                        "source": spec.path,
                                        "stage": f"archive_member:{i}",
                                        "error": f"{type(error).__name__}: {error}",
                                    }
                                )
        except Exception as error:
            failures.append(
                {
                    "source": spec.path,
                    "stage": "parser",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    for statement in selection.statements:
        anchors = [
            o
            for o in by_source[statement.path]
            if o.locator == statement.locator and statement.contains in str(o.raw.get("text", ""))
        ]
        if len(anchors) != 1:
            failures.append(
                {
                    "source": statement.path,
                    "stage": "statement",
                    "error": f"Expected unique text anchor at {statement.locator}",
                }
            )
            continue
        payload: dict[str, Any] = {
            "subject": statement.subject,
            "field": statement.field,
            "value": statement.value,
            "original_units": statement.original_units,
            "context": statement.context,
            "observation_ids": [anchors[0].observation_id],
        }
        bundle.assertions.append(Assertion(assertion_id=stable_id("assert", payload), **payload))
    interval_reports = []
    for link in selection.candidate_links:
        source, target = by_source[link.recording], by_source[link.export]
        roots = (
            [o for o in source if o.kind == "artifact"],
            [o for o in target if o.kind == "artifact"],
        )
        if not all(roots):
            failures.append(
                {
                    "source": link.recording,
                    "stage": "candidate_link",
                    "error": "Missing source/target observations",
                }
            )
            continue
        link_payload = {
            "subject": link.recording,
            "field": "candidate_export",
            "value": link.export,
            "context": {"selection_rationale": link.reason},
            "observation_ids": [roots[0][0].observation_id, roots[1][0].observation_id],
            "evidence_class": "WEAK",
        }
        link_assertion = Assertion.model_validate(
            {"assertion_id": stable_id("assert", link_payload), **link_payload}
        )
        bundle.assertions.append(link_assertion)
        edge = Relationship(
            edge_id=stable_id(
                "edge", [roots[0][0].observation_id, roots[1][0].observation_id, "candidate_export"]
            ),
            source_observation_id=roots[0][0].observation_id,
            target_observation_id=roots[1][0].observation_id,
            relation="candidate_export",
            assertion_ids=[link_assertion.assertion_id],
            notes=link.reason,
        )
        bundle.relationships.append(edge)
        interval_reports.append(
            {"edge_id": edge.edge_id, "comparison": interval_comparison(source, target)}
        )
    for edge in bundle.relationships:
        edge.alternative_edge_ids = [
            other.edge_id
            for other in bundle.relationships
            if other.edge_id != edge.edge_id
            and (
                other.source_observation_id == edge.source_observation_id
                or other.target_observation_id == edge.target_observation_id
            )
        ]
    duplicate_groups: dict[str, list[Artifact]] = defaultdict(list)
    for art in bundle.artifacts:
        duplicate_groups[art.sha256].append(art)
    root_obs = {
        o.artifact_id: o.observation_id for o in bundle.observations if o.kind == "artifact"
    }
    for group in duplicate_groups.values():
        for a, b in zip(group, group[1:], strict=False):
            bundle.relationships.append(
                Relationship(
                    edge_id=stable_id("edge", [a.artifact_id, b.artifact_id, "byte_identical"]),
                    source_observation_id=root_obs[a.artifact_id],
                    target_observation_id=root_obs[b.artifact_id],
                    relation="byte_identical_artifact",
                    evidence_class="EXACT",
                    notes="Artifact equivalence only; dependent evidence, no canonical copy selected.",
                )
            )
    # Every inspected input is re-read even if parsing failed; preserve both results.
    verification = []
    second = Hasher(HashPolicy(mode="full"))
    for entry in manifest:
        path = paths[entry["source_path"]]
        try:
            digest, status = second.compute(path)
            after = signature(path)
            unchanged = (
                status == "hashed" and digest == entry.get("sha256") and after == entry["before"]
            )
            verification.append(
                {
                    "source_path": entry["source_path"],
                    "after_sha256": digest,
                    "after": after,
                    "unchanged": unchanged,
                }
            )
            if not unchanged:
                raise ValueError("Source changed across extraction")
        except (OSError, ValueError) as error:
            failures.append(
                {"source": entry["source_path"], "stage": "verification", "error": str(error)}
            )
    try:
        EvidenceBundle.model_validate(bundle.model_dump())
    except ValueError as error:
        failures.append({"source": "bundle", "stage": "validation", "error": str(error)})
    write_json(out / "evidence.json", bundle.model_dump(mode="json"))
    try:
        validate_bundle_file(out / "evidence.json")
    except ValueError as error:
        failures.append({"source": "bundle", "stage": "manifest_validation", "error": str(error)})
    write_json(out / "evidence.schema.json", EvidenceBundle.model_json_schema())
    write_json(out / "verification.json", verification)
    write_json(out / "interval_comparisons.json", interval_reports)
    write_json(out / "failures.json", failures)
    conflicts: list[dict[str, Any]] = [
        {
            "conflict_id": stable_id("decision", x),
            "status": "UNRESOLVED",
            "decision": x,
            "assertion_ids": [],
            "observation_ids": [],
        }
        for x in selection.unresolved_decisions
    ]
    assertions_by_context: dict[str, list[Assertion]] = defaultdict(list)
    for assertion in bundle.assertions:
        assertions_by_context[
            canonical_json([assertion.subject, assertion.field, assertion.context])
        ].append(assertion)
        if assertion.context.get("leading_zero_risk"):
            conflicts.append(
                {
                    "conflict_id": stable_id("risk", assertion.assertion_id),
                    "status": "UNRESOLVED",
                    "decision": "Unquoted historical R alias may lose leading zeros; no numeric conversion applied",
                    "assertion_ids": [assertion.assertion_id],
                    "observation_ids": assertion.observation_ids,
                }
            )
    for key, candidates in assertions_by_context.items():
        if len({canonical_json(a.value) for a in candidates}) > 1:
            conflicts.append(
                {
                    "conflict_id": stable_id("conflict", key),
                    "status": "CONFLICT",
                    "decision": "Incompatible historical assertions in the same context; resolve source meaning before promotion",
                    "assertion_ids": [a.assertion_id for a in candidates],
                    "observation_ids": [o for a in candidates for o in a.observation_ids],
                }
            )
    recording_evidence = []
    for spec in selection.sources:
        if spec.kind != "wav":
            continue
        rows = by_source[spec.path]
        timestamp_comparison = timestamp_review(rows)
        if timestamp_comparison["status"] == "CONFLICT":
            conflicts.append(
                {
                    "conflict_id": stable_id("timestamp-conflict", spec.path),
                    "status": "CONFLICT",
                    "decision": "Embedded timestamps disagree under the tested date-convention hypotheses",
                    "observation_ids": timestamp_comparison["observation_ids"],
                    "assertion_ids": [
                        a.assertion_id
                        for a in bundle.assertions
                        if a.subject == spec.path and a.field == "acquisition_timestamp_text"
                    ],
                }
            )
        recording_evidence.append(
            {
                "source_path": spec.path,
                "timestamp_comparison": timestamp_comparison,
                "canonical_animal_id": None,
                "canonical_litter_id": None,
                "canonical_dam_id": None,
                "acquisition_timestamp_observation_ids": [
                    o.observation_id for o in rows if o.kind == "timestamp_text"
                ],
                "chunk_observation_ids": [o.observation_id for o in rows if o.kind == "riff_chunk"],
                "label_observation_ids": [o.observation_id for o in rows if o.kind == "label_line"],
                "candidate_export_edge_ids": [
                    e.edge_id
                    for e in bundle.relationships
                    if e.relation == "candidate_export"
                    and e.source_observation_id in {o.observation_id for o in rows}
                ],
                "recording_role": "unresolved",
                "full_decode_frame_coverage": None,
                "frames_decoded": 0,
                "prior_exposure": "metadata and labels inspected in this run",
            }
        )
    write_json(out / "recording_evidence.json", recording_evidence)
    write_json(out / "conflict_ledger.json", conflicts)
    coverage = {
        "schema_version": 2,
        "parser_version": PARSER_VERSION,
        "manifest_sha256": manifest_hash,
        "source_files_selected": len(selection.sources),
        "source_bytes": sum(e["before"]["size_bytes"] for e in manifest),
        "hash_bytes_read": first.bytes_read + second.bytes_read,
        "source_verification_passed": len(verification) == len(manifest)
        and all(v["unchanged"] for v in verification),
        "artifacts": len(bundle.artifacts),
        "observations": len(bundle.observations),
        "assertions": len(bundle.assertions),
        "observation_kinds": dict(Counter(o.kind for o in bundle.observations)),
        "duplicate_groups": [
            [a.artifact_id for a in group] for group in duplicate_groups.values() if len(group) > 1
        ],
        "workbooks": [
            {"artifact_id": o.artifact_id, "locator": o.locator, **o.raw}
            for o in bundle.observations
            if o.kind == "workbook_coverage"
        ],
        "canonical_biological_assignments": 0,
        "human_reviews": 0,
        "full_dataset_freeze": False,
        "failure_count": len(failures),
        "success": not failures,
        "prior_exposure": [s.path for s in selection.sources if s.kind == "wav"],
        "limits": [
            "Selected inputs only; no full freeze or backup",
            "No audio decoding or detector assessment",
            "No identity, sex, day or unit confirmations",
            "PDF figures require visual review; lexical R subset only",
            "Missing weights/dams/allocation/access evidence remains unresolved",
        ],
    }
    write_json(out / "coverage.json", coverage)
    (out / "review.md").write_text(
        "# M0.1 evidence extraction review\n\n"
        f"Status: {'extracted; human review pending' if not failures else 'FAILED; retain partial observations and inspect failures.json'}.\n\n"
        f"Selected files: {len(selection.sources)}. Observations: {len(bundle.observations)}. Candidate assertions: {len(bundle.assertions)}. Canonical assignments: 0.\n\n"
        "Start with coverage.json, conflict_ledger.json and interval_comparisons.json. evidence.json retains hashes, locators, raw values, aliases and lineage. No summary rows count as events. Unit hypotheses in interval comparisons do not authorize conversion.\n\n"
        "source_manifest.json and verification.json record before/after hashes and signatures. This is a bounded immutability check, not a collection freeze. run.json records code/config hashes; selection.json and protocol_snapshot.md preserve inputs. All selected WAVs are logged as prior exposure.\n\n"
        "## Decisions requiring evidence\n\n"
        + "\n".join(f"- {x}" for x in selection.unresolved_decisions)
        + "\n"
    )
    record_run(out, "evidence-pilot", config, inputs)
    return coverage


def validate_bundle_file(path: Path) -> EvidenceBundle:
    """Validate a v2 sidecar and its exact sibling manifest, without reading sources."""
    bundle = EvidenceBundle.model_validate_json(path.read_bytes())
    manifest_path = path.parent / "source_manifest.json"
    if sha256_file(manifest_path) != bundle.manifest_sha256:
        raise ValueError("Source manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    by_path = {r["source_path"]: r for r in manifest}
    if len(by_path) != len(manifest):
        raise ValueError("Duplicate source manifest path")
    for art in bundle.artifacts:
        row = by_path.get(art.source_path)
        if row is None or row.get("error"):
            raise ValueError("Artifact lacks valid manifest entry")
        if art.member_name is None:
            if art.sha256 != row["sha256"] or art.size_bytes != row["before"]["size_bytes"]:
                raise ValueError("Artifact/manifest hash or size mismatch")
        elif art.container_sha256 != row["sha256"]:
            raise ValueError("Archive container/manifest mismatch")
        if art.member_name is not None:
            containers = {
                a.artifact_id
                for a in bundle.artifacts
                if a.source_path == art.source_path and a.member_name is None
            }
            entries = [
                o
                for o in bundle.observations
                if o.artifact_id in containers
                and o.kind == "archive_entry"
                and o.raw.get("member_index") == art.member_index
            ]
            if (
                len(entries) != 1
                or entries[0].raw.get("member_name") != art.member_name
                or entries[0].raw.get("size_bytes") != art.size_bytes
            ):
                raise ValueError("Archive member lacks matching directory provenance")
    return bundle
