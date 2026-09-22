"""Compare historical metadata derivatives without resolving biological identities."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from rat_usv.config import SourceConfig
from rat_usv.evidence.pilot import validate_bundle_file
from rat_usv.evidence.review import fenced
from rat_usv.provenance.io import record_run, sha256_file, write_json
from rat_usv.schemas.evidence import EvidenceBundle, Observation, stable_id

VERSION = "metadata-reconciliation/1"
MAX_FILE_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
MAX_CANDIDATE_EDGES = 100000
READ_EXCEL = re.compile(r'^\s*([Dd]iena_([0-9]+))\s*<-\s*(?:readxl::)?read_excel\(\s*"([^"\n]+)"')
DEPENDENCY = re.compile(
    r"\b(?:library|require|readRDS|saveRDS|read_rds|write_rds|load|save|source|"
    r"system|system2|install\.packages|download\.file|setwd|dir_ls|read_excel|read\.csv)\s*\("
)


def snapshot_run(
    origin: Path, target: Path, command: str, names: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Snapshot exact input bytes, then verify against their supplied upstream run."""
    target.mkdir()
    digests = {}
    total = 0
    for name in ("run.json", *names):
        path = origin / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required regular derivative missing or symlinked: {name}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
            raise ValueError("Derivative snapshot byte budget exceeded")
        with path.open("rb") as handle:
            data = handle.read(min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total) + 1)
        if len(data) != size:
            raise ValueError("Derivative size changed during snapshot")
        total += len(data)
        (target / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    run: dict[str, Any] = json.loads((target / "run.json").read_bytes())
    if run.get("command") != command:
        raise ValueError("Unexpected upstream command")
    outputs = {row["path"]: row["sha256"] for row in run["outputs"]}
    if len(outputs) != len(run["outputs"]):
        raise ValueError("Duplicate upstream output path")
    if any(outputs.get(name) != digests[name] for name in names):
        raise ValueError("Upstream derivative digest mismatch")
    return run, digests


def script_code_lines(bundle: EvidenceBundle) -> Iterator[Observation]:
    """Narrow line/fence filter, not an R parser or evaluator."""
    by_artifact: dict[str, list[Observation]] = defaultdict(list)
    for obs in bundle.observations:
        if obs.kind == "script_line":
            by_artifact[obs.artifact_id].append(obs)
    for artifact in bundle.artifacts:
        rmd = (artifact.member_name or artifact.source_path).lower().endswith(".rmd")
        in_code = not rmd
        for obs in sorted(
            by_artifact[artifact.artifact_id], key=lambda o: int(o.locator.removeprefix("line:"))
        ):
            text = str(obs.raw["text"]).lstrip()
            if rmd and text.startswith("```"):
                in_code = bool(re.match(r"```\{r(?:[\s,}]|$)", text))
                continue
            if in_code and not text.startswith("#"):
                yield obs


def path_tokens(raw: str) -> list[str]:
    """Lexical separators only; never resolve or open a historical path."""
    return [part for part in raw.replace("\\", "/").split("/") if part]


def calendar_deltas(first: str, second: str) -> list[int]:
    """Possible forward gaps in MM.DD tokens under at most one year rollover.

    Synthetic years cover leap at the start, leap at the end, and neither.
    They are arithmetic devices; no calendar year is assigned to the data.
    """
    if not all(re.fullmatch(r"\d{2}\.\d{2}", token) for token in (first, second)):
        return []
    a, b = tuple(map(int, first.split("."))), tuple(map(int, second.split(".")))
    deltas = set()
    for year in (1999, 2000, 2001):
        try:
            start = date(year, *a)
            end = date(year + int(b < a), *b)
        except ValueError:
            continue
        deltas.add((end - start).days)
    return sorted(deltas)


def reconcile_records(
    bundle: EvidenceBundle, entities: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    artifacts = {a.artifact_id: a for a in bundle.artifacts}
    observations = {o.observation_id: o for o in bundle.observations}
    if len({e["entity_id"] for e in entities}) != len(entities):
        raise ValueError("Duplicate search entity ID")
    exports: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        if entity["status"] == "searched_cells_formulas_and_caches":
            parts = path_tokens(entity["member_name"] or entity["source_path"])
            if len(parts) >= 3:
                exports[tuple(parts[-3:])].append(entity)

    def located(obs: Observation) -> dict[str, Any]:
        return {
            "observation": obs.model_dump(mode="json"),
            "artifact": artifacts[obs.artifact_id].model_dump(mode="json"),
        }

    references: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    candidate_edges = 0
    code_lines = list(script_code_lines(bundle))
    code_ids = {o.observation_id for o in code_lines}
    non_code_inputs = [
        {
            "example_id": stable_id("non-code-input", obs.observation_id),
            "evidence": located(obs),
            "status": "outside_recognized_R_code; retained without path comparison",
        }
        for obs in bundle.observations
        if obs.kind == "script_line"
        and obs.observation_id not in code_ids
        and READ_EXCEL.match(str(obs.raw["text"]))
    ]
    for obs in code_lines:
        text = str(obs.raw["text"])
        calls = DEPENDENCY.findall(text)
        if calls:
            dependencies.append(
                {
                    "dependency_line_id": stable_id("dependency-line", obs.observation_id),
                    "lexical_call_tokens": calls,
                    "evidence": located(obs),
                    "executed": False,
                    "coverage": "lexical candidates only; strings and multiline calls need review",
                }
            )
        match = READ_EXCEL.match(text)
        if match is None:
            continue
        parts = path_tokens(match[3])
        usable = len(parts) >= 3 and not any(p in {".", ".."} for p in parts)
        candidates = exports.get(tuple(parts[-3:]), []) if usable else []
        candidate_edges += len(candidates)
        if candidate_edges > MAX_CANDIDATE_EDGES:
            raise ValueError("Candidate edge budget exceeded")
        hashes = {e["sha256"] for e in candidates}
        references.append(
            {
                "reference_id": stable_id("historical-input", obs.observation_id),
                "variable_token": match[1],
                "day_label_token": match[2],
                "quoted_path": match[3],
                "tail_three_tokens": parts[-3:] if usable else None,
                "status": "insufficient_path_context"
                if not usable
                else "no_suffix_candidate"
                if not candidates
                else "candidates_different_content"
                if len(hashes) > 1
                else "candidates_one_content_hash",
                "evidence": located(obs),
                "candidate_entities": candidates,
                "distinct_candidate_hashes": len(hashes),
                "match_rule": "exact case-sensitive final three path components; all copies retained",
                "canonical_animal_id": None,
                "recording_session_id": None,
            }
        )
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        if row["tail_three_tokens"] is not None:
            contexts[
                row["evidence"]["artifact"]["artifact_id"], row["variable_token"].split("_")[0]
            ].append(row)
    mixed = []
    for (artifact_id, prefix), rows in contexts.items():
        tails = {tuple(r["tail_three_tokens"][:2]) for r in rows}
        if len(tails) > 1:
            mixed.append(
                {
                    "finding_id": stable_id("mixed-input-context", [artifact_id, prefix]),
                    "artifact_id": artifact_id,
                    "variable_prefix": prefix,
                    "path_contexts": [list(t) for t in sorted(tails)],
                    "reference_ids": [r["reference_id"] for r in rows],
                    "interpretation": "mixed path contexts in one script; may be intentional examples or an error",
                    "status": "UNRESOLVED",
                }
            )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for assertion in bundle.assertions:
        if assertion.field != "historical_day_label":
            continue
        if len(assertion.observation_ids) != 1:
            raise ValueError("Day-label audit requires one explicit source line per assertion")
        obs = observations[assertion.observation_ids[0]]
        if not isinstance(assertion.value, str) or not assertion.value.isdecimal():
            raise ValueError("Historical day label must retain its original decimal string")
        key = (
            obs.artifact_id,
            str(assertion.context["binding"]),
            str(assertion.context["recode_block"]),
        )
        groups[key].append(
            {"assertion": assertion.model_dump(mode="json"), "evidence": located(obs)}
        )
    recodes = []
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r["evidence"]["observation"]["locator"].removeprefix("line:")))
        comparisons = []
        for first, second in zip(rows, rows[1:], strict=False):
            a, b = first["assertion"], second["assertion"]
            gaps = calendar_deltas(a["context"]["date_token"], b["context"]["date_token"])
            delta = int(b["value"]) - int(a["value"])
            status = (
                "invalid_date_token"
                if not gaps
                else "non_increasing_day_labels"
                if delta <= 0
                else "delta_disagreement"
                if delta not in gaps
                else "leap_year_ambiguous"
                if len(gaps) > 1
                else "compatible_under_forward_calendar_hypothesis"
            )
            comparisons.append(
                {
                    "comparison_id": stable_id("day-delta", [a["assertion_id"], b["assertion_id"]]),
                    "assertion_ids": [a["assertion_id"], b["assertion_id"]],
                    "day_label_delta": delta,
                    "possible_calendar_deltas": gaps,
                    "status": status,
                }
            )
        recodes.append(
            {
                "group_id": stable_id("recode-group", key),
                "artifact_id": key[0],
                "binding": key[1],
                "recode_block": key[2],
                "rows": rows,
                "adjacent_comparisons": comparisons,
                "calendar_hypothesis": "MM.DD, forward order, at most one rollover; year unknown",
                "postnatal_age_assigned": False,
            }
        )
    return {
        "input_references": references,
        "mixed_input_contexts": mixed,
        "day_recode_groups": recodes,
        "dependency_lines": dependencies,
        "non_code_input_examples": non_code_inputs,
    }


def reconcile_metadata(
    config: SourceConfig, evidence: Path, search_entities: Path, out: Path
) -> dict[str, Any]:
    evidence, search_entities = evidence.resolve(strict=True), search_entities.resolve(strict=True)
    if evidence.name != "evidence.json" or search_entities.name != "entities.json":
        raise ValueError("Expected evidence.json and metadata-search entities.json")
    out = out.resolve(strict=True)
    if (
        out == config.output_root
        or not out.is_relative_to(config.output_root)
        or any(out.is_relative_to(p.parent) for p in (evidence, search_entities))
        or any(out.iterdir())
    ):
        raise ValueError("Reconciliation requires a separate new empty output run")
    evidence_dir, search_dir = out / "evidence_inputs", out / "search_inputs"
    evidence_run, evidence_hashes = snapshot_run(
        evidence.parent,
        evidence_dir,
        "evidence-pilot",
        (
            "evidence.json",
            "source_manifest.json",
            "protocol_snapshot.md",
            "coverage.json",
            "failures.json",
        ),
    )
    search_run, search_hashes = snapshot_run(
        search_entities.parent,
        search_dir,
        "metadata-search",
        ("entities.json", "summary.json", "failures.json", "policy.json"),
    )
    bundle = validate_bundle_file(evidence_dir / "evidence.json")
    coverage = json.loads((evidence_dir / "coverage.json").read_bytes())
    search_summary = json.loads((search_dir / "summary.json").read_bytes())
    if (
        bundle.protocol_sha256 != evidence_hashes["protocol_snapshot.md"]
        or coverage.get("success") is not True
        or coverage.get("source_verification_passed") is not True
        or json.loads((evidence_dir / "failures.json").read_bytes()) != []
        or search_summary.get("source_verification_passed") is not True
    ):
        raise ValueError("Reconciliation requires source-verified inputs and successful evidence")
    if not (
        evidence_run["config"]["source_root"]
        == search_run["config"]["source_root"]
        == str(config.source_root)
    ):
        raise ValueError("Input runs refer to different source roots")
    entities = json.loads((search_dir / "entities.json").read_bytes())
    outer = {e["source_path"]: e for e in entities if e["member_name"] is None}
    for artifact in bundle.artifacts:
        source_digest = artifact.container_sha256 or artifact.sha256
        if outer.get(artifact.source_path, {}).get("sha256") != source_digest:
            raise ValueError("Evidence/search source hash mismatch or source missing")
    result = reconcile_records(bundle, entities)
    for name, rows in result.items():
        write_json(out / f"{name}.json", rows)
    comparisons = [c for g in result["day_recode_groups"] for c in g["adjacent_comparisons"]]
    summary = {
        "version": VERSION,
        "evidence_run": evidence_run["run_id"],
        "search_run": search_run["run_id"],
        "protocol_sha256": bundle.protocol_sha256,
        "source_manifest_sha256": bundle.manifest_sha256,
        "input_hashes": {"evidence": evidence_hashes, "search": search_hashes},
        "input_reference_statuses": dict(Counter(r["status"] for r in result["input_references"])),
        "mixed_input_contexts": len(result["mixed_input_contexts"]),
        "recode_groups": len(result["day_recode_groups"]),
        "recode_assertions": sum(len(g["rows"]) for g in result["day_recode_groups"]),
        "day_comparison_statuses": dict(Counter(c["status"] for c in comparisons)),
        "dependency_lines": len(result["dependency_lines"]),
        "non_code_input_examples": len(result["non_code_input_examples"]),
        "upstream_search_success": search_summary["success"],
        "upstream_search_failures": json.loads((search_dir / "failures.json").read_bytes()),
        "raw_source_files_opened": 0,
        "new_canonical_assignments": 0,
        "new_review_decisions": 0,
        "limits": [
            "Exact path suffix equality is a candidate artifact correspondence, not animal identity",
            "Compatible date arithmetic does not validate age, date convention or recording attendance",
            "Script extraction is lexical and incomplete; examples and repeated assignments need review",
            "All copies remain; counts are not independent animals or corroborations",
            "Upstream search gaps remain explicit; no global absence claim",
        ],
    }
    write_json(out / "summary.json", summary)
    write_json(out / "input_manifest.json", summary["input_hashes"])
    with (out / "review.md").open("x", encoding="utf-8") as handle:
        handle.write(
            "# Historical metadata reconciliation\n\n"
            "Candidate comparisons only. All scientific decisions remain unresolved.\n\n"
            "[Input references](input_references.json) retain every matching export copy and source "
            "line. [Mixed contexts](mixed_input_contexts.json) identify scripts requiring review. "
            "[Date recodes](day_recode_groups.json) retain every scoped assertion and comparison. "
            "[Dependency lines](dependency_lines.json) are lexical observations, not executed code. "
            "[Non-code examples](non_code_input_examples.json) retain unmatched-scope input lines.\n\n"
            + fenced(summary)
        )
    inputs = [p for folder in (evidence_dir, search_dir) for p in sorted(folder.iterdir())]
    record_run(out, "reconcile-metadata", config, inputs)
    for folder, expected in ((evidence_dir, evidence_hashes), (search_dir, search_hashes)):
        if any(sha256_file(folder / name) != digest for name, digest in expected.items()):
            raise ValueError("Derivative snapshot changed during reconciliation")
    return summary
