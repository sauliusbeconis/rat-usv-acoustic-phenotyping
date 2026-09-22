"""Versioned evidence sidecars. These records never implicitly populate canonical registries."""

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from rat_usv.schemas.registry import EvidenceClass

SCHEMA_VERSION: Literal[2] = 2
PARSER_VERSION: Literal["m0.1/2"] = "m0.1/2"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, Field(min_length=1)]


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def stable_id(kind: str, value: object) -> str:
    return kind + "-" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Artifact(Record):
    artifact_id: Nonempty
    source_path: Nonempty
    sha256: Digest
    size_bytes: int = Field(ge=0)
    member_name: str | None = None
    member_index: int | None = Field(default=None, ge=0)
    container_sha256: Digest | None = None
    lineage: Nonempty
    independence_review_id: str | None = None

    @model_validator(mode="after")
    def member_provenance(self) -> Self:
        values = (self.member_name, self.member_index, self.container_sha256)
        if any(v is not None for v in values) and not all(v is not None for v in values):
            raise ValueError("Archive member needs name, ordinal and container hash")
        for value in (self.source_path, self.member_name):
            if value is not None:
                path = PurePosixPath(value)
                if not path.parts or path.is_absolute() or ".." in path.parts:
                    raise ValueError("Artifact paths must be source-relative without traversal")
        return self


class Observation(Record):
    observation_id: Nonempty
    artifact_id: Nonempty
    locator: Nonempty
    kind: Nonempty
    raw: dict[str, JsonValue]
    parser_version: Nonempty = PARSER_VERSION
    manifest_sha256: Digest


class Assertion(Record):
    assertion_id: Nonempty
    subject: Nonempty  # scoped candidate token, never a canonical biological ID
    field: Nonempty
    value: JsonValue
    original_units: str | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)
    observation_ids: list[str] = Field(min_length=1)
    derivation_ids: list[str] = Field(default_factory=list)
    evidence_class: EvidenceClass = "MODERATE"
    status: Literal["unresolved", "confirmed", "conflicting", "rejected", "superseded"] = (
        "unresolved"
    )
    canonical_id: str | None = None
    supersedes: str | None = None
    review_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def prerequisites(self) -> Self:
        if self.field == "weight" and not self.context.get("measurement_occasion"):
            raise ValueError("Weight assertion requires measurement_occasion")
        if self.evidence_class == "CONFLICT" and self.status not in {
            "conflicting",
            "rejected",
            "superseded",
        }:
            raise ValueError("Conflict must remain explicit")
        if self.canonical_id is not None and self.status != "confirmed":
            raise ValueError("Unconfirmed assertion cannot assign canonical identity")
        if self.status == "confirmed" and (
            not self.review_ids or self.evidence_class not in {"EXACT", "STRONG"}
        ):
            raise ValueError("Confirmation requires review and exact/strong evidence")
        return self


class Relationship(Record):
    edge_id: Nonempty
    source_observation_id: Nonempty
    target_observation_id: Nonempty
    relation: Nonempty
    assertion_ids: list[str] = Field(default_factory=list)
    evidence_class: EvidenceClass = "WEAK"
    alternative_edge_ids: list[str] = Field(default_factory=list)
    review_ids: list[str] = Field(default_factory=list)
    canonical_id: None = None
    notes: str = ""


class ReviewDecision(Record):
    review_id: Nonempty
    actor: Nonempty
    reviewed_utc: str
    protocol_sha256: Digest
    target_ids: list[str] = Field(min_length=1)
    disposition: Literal["confirm", "reject", "supersede", "independence"]
    rationale: Nonempty
    supporting_ids: list[str] = Field(min_length=1)
    opposing_ids: list[str] = Field(default_factory=list)
    competing_dispositions: dict[str, str] = Field(default_factory=dict)
    supersedes: str | None = None

    @model_validator(mode="after")
    def timestamp(self) -> Self:
        date = datetime.fromisoformat(self.reviewed_utc.replace("Z", "+00:00"))
        offset = date.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Review timestamp must explicitly use UTC")
        if not set(self.opposing_ids) <= self.competing_dispositions.keys():
            raise ValueError("Every opposing assertion requires explicit disposition")
        if (
            not self.actor.strip()
            or not self.rationale.strip()
            or any(not value.strip() for value in self.competing_dispositions.values())
        ):
            raise ValueError("Review actor, rationale and dispositions cannot be blank")
        return self


class EvidenceBundle(Record):
    schema_version: Literal[2] = SCHEMA_VERSION
    manifest_sha256: Digest
    protocol_sha256: Digest
    artifacts: list[Artifact] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    reviews: list[ReviewDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def graph_valid(self) -> Self:
        def index(rows: list[Record], key: str) -> dict[str, Record]:
            result = {getattr(row, key): row for row in rows}
            if len(result) != len(rows):
                raise ValueError(f"Duplicate {key}")
            return result

        artifacts = {a.artifact_id: a for a in self.artifacts}
        observations = {o.observation_id: o for o in self.observations}
        assertions = {a.assertion_id: a for a in self.assertions}
        edges = {e.edge_id: e for e in self.relationships}
        reviews = {r.review_id: r for r in self.reviews}
        for rows, key in (
            (self.artifacts, "artifact_id"),
            (self.observations, "observation_id"),
            (self.assertions, "assertion_id"),
            (self.relationships, "edge_id"),
            (self.reviews, "review_id"),
        ):
            index(list(rows), key)
        all_ids = set(artifacts) | set(observations) | set(assertions) | set(edges) | set(reviews)
        if len(all_ids) != len(artifacts) + len(observations) + len(assertions) + len(edges) + len(
            reviews
        ):
            raise ValueError("Evidence IDs must be unique across record types")

        def require(ids: list[str], known: Iterable[str]) -> None:
            if not set(ids) <= set(known):
                raise ValueError("Unknown evidence reference")

        for obs in self.observations:
            require([obs.artifact_id], artifacts)
            if obs.manifest_sha256 != self.manifest_sha256:
                raise ValueError("Observation manifest mismatch")
            expected = stable_id(
                "obs", [obs.artifact_id, obs.locator, obs.kind, obs.raw, obs.parser_version]
            )
            if obs.observation_id != expected:
                raise ValueError("Observation content ID mismatch")
        for review in self.reviews:
            require(review.target_ids + review.supporting_ids + review.opposing_ids, all_ids)
            require(review.opposing_ids, assertions)
            require(list(review.competing_dispositions), assertions)
            if review.protocol_sha256 != self.protocol_sha256:
                raise ValueError("Review protocol mismatch")
        for art in self.artifacts:
            if art.independence_review_id:
                require([art.independence_review_id], reviews)
                review = reviews[art.independence_review_id]
                if review.disposition != "independence" or art.artifact_id not in review.target_ids:
                    raise ValueError("Independence review does not cover artifact")
        for edge in self.relationships:
            require([edge.source_observation_id, edge.target_observation_id], observations)
            require(edge.assertion_ids, assertions)
            require(edge.alternative_edge_ids, edges)
            require(edge.review_ids, reviews)
        confirmed: dict[str, set[str]] = {}
        for assertion in self.assertions:
            require(assertion.observation_ids, observations)
            require(assertion.derivation_ids, assertions)
            require(assertion.review_ids, reviews)
            if assertion.supersedes:
                require([assertion.supersedes], assertions)
                prior = assertions[assertion.supersedes]
                if (prior.subject, prior.field, prior.context) != (
                    assertion.subject,
                    assertion.field,
                    assertion.context,
                ):
                    raise ValueError("Supersession subject/context mismatch")
            for ref in assertion.derivation_ids:
                if assertions[ref].subject != assertion.subject:
                    raise ValueError("Derivation evidence subject mismatch")
            if assertion.status == "confirmed":
                key = canonical_json([assertion.subject, assertion.field, assertion.context])
                confirmed.setdefault(key, set()).add(canonical_json(assertion.value))
                competing = {
                    other.assertion_id
                    for other in self.assertions
                    if (other.subject, other.field, other.context)
                    == (assertion.subject, assertion.field, assertion.context)
                    and canonical_json(other.value) != canonical_json(assertion.value)
                }
                confirmations = [
                    reviews[r]
                    for r in assertion.review_ids
                    if reviews[r].disposition == "confirm"
                    and assertion.assertion_id in reviews[r].target_ids
                    and set(assertion.observation_ids) <= set(reviews[r].supporting_ids)
                    and competing <= reviews[r].competing_dispositions.keys()
                ]
                if not confirmations:
                    raise ValueError("Confirmation lacks supporting/competing review")
                sources = [
                    artifacts[observations[o].artifact_id] for o in assertion.observation_ids
                ]
                if assertion.evidence_class == "STRONG" and (
                    len({a.lineage for a in sources}) < 2
                    or len({a.sha256 for a in sources}) < 2
                    or any(not a.independence_review_id for a in sources)
                ):
                    raise ValueError(
                        "Copied or unreviewed evidence is not independent corroboration"
                    )
        if any(len(v) > 1 for v in confirmed.values()):
            raise ValueError("Contradictory confirmed assertions in the same context")
        for nodes in (assertions, reviews):
            for start in nodes:
                seen: set[str] = set()
                current: str | None = start
                while current:
                    if current in seen:
                        raise ValueError("Supersession cycle")
                    seen.add(current)
                    require([current], nodes)
                    current = nodes[current].supersedes
        return self
