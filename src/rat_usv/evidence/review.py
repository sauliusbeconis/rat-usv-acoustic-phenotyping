"""Prepare traceable review material from existing derivatives, without adjudication."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rat_usv.config import SourceConfig
from rat_usv.evidence.pilot import validate_bundle_file
from rat_usv.provenance.io import record_run, sha256_file, write_json
from rat_usv.schemas.evidence import (
    Artifact,
    Assertion,
    Digest,
    EvidenceBundle,
    Nonempty,
    Observation,
    Record,
    Relationship,
)

PACKET_VERSION = "m0.3/1"
INPUT_NAMES = (
    "run.json",
    "evidence.json",
    "source_manifest.json",
    "conflict_ledger.json",
    "protocol_snapshot.md",
    "coverage.json",
    "failures.json",
    "code_snapshot.json",
)
MAX_INPUT_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2


class LedgerEntry(Record):
    conflict_id: Nonempty
    status: Literal["UNRESOLVED", "CONFLICT"]
    decision: Nonempty
    assertion_ids: list[str]
    observation_ids: list[str]


class LocatedObservation(Record):
    observation: Observation
    artifact: Artifact


class LocatedAssertion(Record):
    assertion: Assertion
    evidence: list[LocatedObservation]


class LocatedRelationship(Record):
    relationship: Relationship
    source: LocatedObservation
    target: LocatedObservation


class ReviewItem(Record):
    ledger_entry: LedgerEntry
    evidence_links: Literal["present", "missing_in_ledger"]
    assertions: list[LocatedAssertion]
    observations: list[LocatedObservation]


class RecordedOutput(Record):
    path: Nonempty
    sha256: Digest


class UpstreamRun(BaseModel):
    # Other run metadata is retained byte-for-byte in the snapshot.
    model_config = ConfigDict(strict=True)
    run_id: Nonempty
    command: Literal["evidence-pilot"]
    code_config_sha256: Digest
    outputs: list[RecordedOutput] = Field(min_length=1)


def expand_review(
    bundle: EvidenceBundle, entries: list[LedgerEntry]
) -> tuple[list[ReviewItem], list[LocatedAssertion], list[LocatedRelationship]]:
    """Preserve every row/reference, including repeated references and dependent copies."""
    if len({e.conflict_id for e in entries}) != len(entries):
        raise ValueError("Duplicate conflict ID")
    artifacts = {a.artifact_id: a for a in bundle.artifacts}
    observations = {
        o.observation_id: LocatedObservation(observation=o, artifact=artifacts[o.artifact_id])
        for o in bundle.observations
    }
    assertions = {
        a.assertion_id: LocatedAssertion(
            assertion=a, evidence=[observations[o] for o in a.observation_ids]
        )
        for a in bundle.assertions
    }
    items = []
    for entry in entries:
        if not set(entry.assertion_ids) <= assertions.keys():
            raise ValueError(f"Unknown ledger assertion reference: {entry.conflict_id}")
        if not set(entry.observation_ids) <= observations.keys():
            raise ValueError(f"Unknown ledger observation reference: {entry.conflict_id}")
        items.append(
            ReviewItem(
                ledger_entry=entry,
                evidence_links="present"
                if entry.assertion_ids or entry.observation_ids
                else "missing_in_ledger",
                assertions=[assertions[a] for a in entry.assertion_ids],
                observations=[observations[o] for o in entry.observation_ids],
            )
        )
    relationships = [
        LocatedRelationship(
            relationship=edge,
            source=observations[edge.source_observation_id],
            target=observations[edge.target_observation_id],
        )
        for edge in bundle.relationships
    ]
    return items, list(assertions.values()), relationships


def fenced(value: object) -> str:
    """Keep source text inert in Markdown, including embedded Markdown fences."""
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", body)), default=0))
    return f"{fence}json\n{body}\n{fence}\n"


def render_review(items: list[ReviewItem], summary: dict[str, object]) -> str:
    lines = [
        "# Evidence review packet\n",
        "This is a navigation aid for the snapshotted M0.1 ledger. All decisions retain "
        "their original status and wording, including any later-outdated gate descriptions. "
        "No human review, identity assignment, exclusion or unit conversion is performed.\n",
        "An evidence link locates an assertion; it does not verify its truth or applicability. "
        "Missing links mean the ledger supplied no references, not that no evidence exists. "
        "Counts are records, not independent animals or corroborating sources.\n",
        "[Full packet](review_packet.json) retains raw observations and archive provenance. "
        "[All assertions](assertions.json) includes every context, unit and source locator; "
        "[all relationships](relationships.json) retains alternatives and copied artifacts. "
        "The exact upstream files are preserved in `inputs/`.\n",
        "## Coverage\n",
        fenced(summary),
    ]
    for number, item in enumerate(items, 1):
        entry = item.ledger_entry
        lines.extend(
            [
                f"## Item {number:03d}: {entry.status}\n",
                fenced({"conflict_id": entry.conflict_id, "decision": entry.decision}),
                "Evidence references: " + item.evidence_links + ".\n",
            ]
        )
        for located in item.assertions:
            lines.append(fenced(located.assertion.model_dump(mode="json")))
        # The two lists are kept separate: neither is silently deduplicated.
        for label, references in (
            ("Direct ledger observation references", item.observations),
            ("Assertion observation references", [o for a in item.assertions for o in a.evidence]),
        ):
            if references:
                lines.append(label + ":\n")
                lines.append(
                    fenced(
                        [
                            {
                                "observation_id": r.observation.observation_id,
                                "locator": r.observation.locator,
                                "artifact": r.artifact.model_dump(mode="json"),
                            }
                            for r in references
                        ]
                    )
                )
    return "\n".join(lines)


def review_packet(config: SourceConfig, evidence_path: Path, out: Path) -> dict[str, object]:
    """Validate and snapshot an extraction run; write a new private review derivative."""
    origin = evidence_path.resolve(strict=True)
    if origin.name != "evidence.json":
        raise ValueError("Expected evidence.json in its original extraction run")
    out = out.resolve(strict=True)
    if (
        out == config.output_root
        or not out.is_relative_to(config.output_root)
        or out.is_relative_to(origin.parent)
        or any(out.iterdir())
    ):
        raise ValueError(
            "Review output must be a new empty run under output_root, apart from input"
        )
    snapshot = out / "inputs"
    snapshot.mkdir()
    total = 0
    digests = {}
    for name in INPUT_NAMES:
        path = origin.parent / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required regular input missing or symlinked: {name}")
        size = path.stat().st_size
        if size > MAX_INPUT_BYTES or total + size > MAX_TOTAL_BYTES:
            raise ValueError(f"Review input byte budget exceeded: {name}")
        with path.open("rb") as handle:
            data = handle.read(min(MAX_INPUT_BYTES, MAX_TOTAL_BYTES - total) + 1)
        if len(data) != size:
            raise ValueError(f"Review input size changed during snapshot: {name}")
        total += len(data)
        with (snapshot / name).open("xb") as handle:
            handle.write(data)
        digests[name] = hashlib.sha256(data).hexdigest()

    upstream = UpstreamRun.model_validate_json((snapshot / "run.json").read_bytes())
    recorded = {row.path: row.sha256 for row in upstream.outputs}
    if len(recorded) != len(upstream.outputs):
        raise ValueError("Duplicate upstream output path")
    for name in INPUT_NAMES[1:]:
        if recorded.get(name) != digests[name]:
            raise ValueError(f"Upstream output digest mismatch: {name}")
    bundle = validate_bundle_file(snapshot / "evidence.json")
    if bundle.protocol_sha256 != digests["protocol_snapshot.md"]:
        raise ValueError("Protocol snapshot digest mismatch")
    coverage = json.loads((snapshot / "coverage.json").read_bytes())
    if (
        not isinstance(coverage, dict)
        or coverage.get("success") is not True
        or coverage.get("source_verification_passed") is not True
        or json.loads((snapshot / "failures.json").read_bytes()) != []
    ):
        raise ValueError("Review packet requires a successful, source-verified extraction")
    entries = TypeAdapter(list[LedgerEntry]).validate_json(
        (snapshot / "conflict_ledger.json").read_bytes()
    )
    items, assertions, relationships = expand_review(bundle, entries)
    summary: dict[str, object] = {
        "packet_version": PACKET_VERSION,
        "upstream_run_id": upstream.run_id,
        "upstream_run_sha256": digests["run.json"],
        "upstream_code_config_sha256": upstream.code_config_sha256,
        "source_evidence_path": str(origin),
        "source_manifest_sha256": bundle.manifest_sha256,
        "protocol_sha256": bundle.protocol_sha256,
        "input_sha256": digests,
        "ledger_items": len(items),
        "ledger_status_counts": dict(Counter(e.status for e in entries)),
        "items_without_evidence_links": sum(i.evidence_links == "missing_in_ledger" for i in items),
        "assertions": len(assertions),
        "relationships": len(relationships),
        "upstream_review_records": len(bundle.reviews),
        "upstream_canonical_assertions": sum(a.canonical_id is not None for a in bundle.assertions),
        "new_review_decisions": 0,
        "new_canonical_assignments": 0,
        "raw_source_files_opened": 0,
        "scientific_readiness_assessed": False,
        "audit_limit": "Hashes check consistency with the supplied run, not independent approval.",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "review_packet.json", [i.model_dump(mode="json") for i in items])
    write_json(out / "assertions.json", [a.model_dump(mode="json") for a in assertions])
    write_json(out / "relationships.json", [e.model_dump(mode="json") for e in relationships])
    with (out / "review.md").open("x", encoding="utf-8") as handle:
        handle.write(render_review(items, summary))
    record_run(out, "review-packet", config, [snapshot / name for name in INPUT_NAMES])
    # Snapshot bytes used above are the input provenance, even if originals later change.
    if any(sha256_file(snapshot / name) != digest for name, digest in digests.items()):
        raise ValueError("Review snapshot changed during packet construction")
    return summary
