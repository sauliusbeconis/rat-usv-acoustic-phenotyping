"""Review and provenance invariants using fabricated evidence only."""

import pytest
from pydantic import ValidationError

from rat_usv.schemas.evidence import (
    Artifact,
    Assertion,
    EvidenceBundle,
    Observation,
    ReviewDecision,
    stable_id,
)

pytestmark = pytest.mark.scientific


def bundle_with_review(field="sex", contexts=None, values=None, copied=False):
    contexts = contexts or [{}, {}]
    values = values or ["female", "male"]
    artifacts, observations, assertions, reviews = [], [], [], []
    for i in range(2):
        digest = "a" * 64 if copied else str(i) * 64
        art = Artifact(
            artifact_id=f"art{i}",
            source_path=f"source{i}.txt",
            sha256=digest,
            size_bytes=1,
            lineage="one_history" if copied else f"history{i}",
        )
        raw = {"text": f"source {i}"}
        obs = Observation(
            observation_id=stable_id("obs", [art.artifact_id, "line:1", "text", raw, "m0.1/2"]),
            artifact_id=art.artifact_id,
            locator="line:1",
            kind="text",
            raw=raw,
            manifest_sha256="b" * 64,
        )
        assertion = Assertion(
            assertion_id=f"assert{i}",
            subject="scope:0001",
            field=field,
            value=values[i],
            context=contexts[i],
            observation_ids=[obs.observation_id],
            status="confirmed",
            evidence_class="EXACT",
            review_ids=[f"review{i}"],
        )
        review = ReviewDecision(
            review_id=f"review{i}",
            actor="synthetic reviewer",
            reviewed_utc="2026-01-01T00:00:00Z",
            protocol_sha256="c" * 64,
            target_ids=[assertion.assertion_id],
            disposition="confirm",
            rationale="Synthetic validation",
            supporting_ids=[obs.observation_id],
            opposing_ids=[f"assert{1 - i}"],
            competing_dispositions={f"assert{1 - i}": "Explicit synthetic disposition"},
        )
        artifacts.append(art)
        observations.append(obs)
        assertions.append(assertion)
        reviews.append(review)
    return dict(
        manifest_sha256="b" * 64,
        protocol_sha256="c" * 64,
        artifacts=artifacts,
        observations=observations,
        assertions=assertions,
        reviews=reviews,
    )


def test_context_specific_weights_do_not_conflict():
    valid = bundle_with_review(
        "weight",
        [{"measurement_occasion": "session1"}, {"measurement_occasion": "session2"}],
        [9.0, 10.0],
    )
    EvidenceBundle(**valid)
    invalid = bundle_with_review("weight", [{"measurement_occasion": "session1"}] * 2, [9.0, 10.0])
    with pytest.raises(ValidationError, match="Contradictory confirmed"):
        EvidenceBundle(**invalid)


def test_confirmed_sex_contradictions_and_missing_review_fail():
    raw = bundle_with_review()
    with pytest.raises(ValidationError, match="Contradictory confirmed"):
        EvidenceBundle(**raw)
    raw = bundle_with_review(values=["female", "female"])
    raw["reviews"][0] = raw["reviews"][0].model_copy(update={"target_ids": ["assert1"]})
    with pytest.raises(ValidationError, match="supporting/competing review"):
        EvidenceBundle(**raw)


def test_copies_cannot_become_strong_corroboration():
    raw = bundle_with_review(values=["female", "female"], copied=True)
    observation_ids = [o.observation_id for o in raw["observations"]]
    raw["assertions"][0] = raw["assertions"][0].model_copy(
        update={"observation_ids": observation_ids, "evidence_class": "STRONG"}
    )
    raw["reviews"][0] = raw["reviews"][0].model_copy(update={"supporting_ids": observation_ids})
    with pytest.raises(ValidationError, match="not independent"):
        EvidenceBundle(**raw)


def test_different_files_also_require_independence_review():
    raw = bundle_with_review(values=["female", "female"])
    ids = [o.observation_id for o in raw["observations"]]
    raw["assertions"][0] = raw["assertions"][0].model_copy(
        update={"observation_ids": ids, "evidence_class": "STRONG"}
    )
    raw["reviews"][0] = raw["reviews"][0].model_copy(update={"supporting_ids": ids})
    with pytest.raises(ValidationError, match="not independent"):
        EvidenceBundle(**raw)


def test_supersession_cycle_and_subject_mismatch():
    raw = bundle_with_review(values=["female", "female"])
    for i in range(2):
        raw["assertions"][i] = raw["assertions"][i].model_copy(
            update={"supersedes": f"assert{1 - i}"}
        )
    with pytest.raises(ValidationError, match="cycle"):
        EvidenceBundle(**raw)
    raw = bundle_with_review(values=["female", "female"])
    raw["assertions"][0] = raw["assertions"][0].model_copy(update={"derivation_ids": ["assert1"]})
    raw["assertions"][1] = raw["assertions"][1].model_copy(update={"subject": "different:0001"})
    with pytest.raises(ValidationError, match="subject mismatch"):
        EvidenceBundle(**raw)


def test_missing_alternative_disposition_and_invalid_review_time():
    raw = bundle_with_review()
    raw["reviews"][0] = raw["reviews"][0].model_copy(update={"competing_dispositions": {}})
    with pytest.raises(ValidationError, match="explicit disposition"):
        EvidenceBundle(**raw)
    review = bundle_with_review()["reviews"][0].model_dump()
    for date in ("2026-01-01", "invalid", "2026-01-01T00:00:00+03:00"):
        with pytest.raises(ValidationError):
            ReviewDecision.model_validate({**review, "reviewed_utc": date})


def test_unknown_observation_fails_and_null_never_zero():
    raw = bundle_with_review(values=[None, None])
    EvidenceBundle(**raw)
    raw["assertions"][0] = raw["assertions"][0].model_copy(update={"observation_ids": ["missing"]})
    with pytest.raises(ValidationError, match="Unknown"):
        EvidenceBundle(**raw)
