"""Whole-decode validation with synthetic PCM only; no private files in CI."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from rat_usv.cli import app
from rat_usv.config import SourceConfig
from rat_usv.provenance.io import new_run, sha256_file, write_json
from rat_usv.qc import full_decode as qc
from rat_usv.schemas.evidence import (
    PARSER_VERSION,
    Artifact,
    EvidenceBundle,
    Observation,
    stable_id,
)


def riff(chunks):
    body = b"WAVE" + b"".join(
        tag + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")
        for tag, data in chunks
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def wave_bytes(values, rate=100, *, extra=False):
    values = np.asarray(values, dtype="<i2")
    channels = values.shape[1] if values.ndim == 2 else 1
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
    chunks = [(b"fmt ", fmt), (b"data", values.tobytes())]
    if extra:
        chunks.insert(0, (b"JUNK", b"x"))
    return riff(chunks)


def inspect(tmp_path, values, policy=None):
    source = tmp_path / "synthetic.wav"
    source.write_bytes(wave_bytes(values))
    before = sha256_file(source)
    blocks, intervals = [], []
    result = qc.decode_pcm16(
        source, policy or qc.DecodePolicy(block_frames=4), blocks.append, intervals.append
    )
    assert before == sha256_file(source)
    return result, blocks, intervals


@pytest.mark.parametrize("block_frames", [1, 2, 4, 7, 65536])
def test_full_decode_exact_integer_counts_and_block_independent_runs(tmp_path, block_frames):
    values = np.array(
        [
            [0, 10],
            [0, 32767],
            [0, -32768],
            [0, 0],
            [0, 0],
            [5, 0],
            [32767, 0],
            [-32768, 0],
            [32766, 1],
            [-32767, 0],
            [0, 0],
        ],
        dtype=np.int16,
    )
    result, blocks, intervals = inspect(
        tmp_path, values, qc.DecodePolicy(block_frames=block_frames)
    )
    assert result["success"] and result["decode_complete"]
    assert result["coverage_fraction"] == 1 and result["decoded_frames"] == 11
    assert result["coverage_intervals"] == [{"start_frame": 0, "stop_frame": 11}]
    assert result["uninspected_intervals"] == [] and result["automatic_exclusion"] is False
    assert sum(b["stop_frame"] - b["start_frame"] for b in blocks) == 11
    assert all(
        a["stop_frame"] == b["start_frame"] for a, b in zip(blocks, blocks[1:], strict=False)
    )
    for c in range(2):
        x = [int(v) for v in values[:, c]]
        channel = result["channels"][c]
        assert channel["total"] == sum(x)
        assert channel["sum_squares"] == sum(v * v for v in x)
        assert channel["rms"] == pytest.approx((sum(v * v for v in x) / len(x)) ** 0.5)
        assert channel["positive_endpoint_samples"] == x.count(32767)
        assert channel["negative_endpoint_samples"] == x.count(-32768)
        assert channel["zero_samples"] == x.count(0)
    observed = {
        (i["channel_index"], i["kind"], i["start_frame"], i["stop_frame"]) for i in intervals
    }
    assert observed == {
        (0, "digital_zero_run", 0, 5),
        (0, "digital_zero_run", 10, 11),
        (0, "pcm_endpoint_run", 6, 8),
        (1, "pcm_endpoint_run", 1, 3),
        (1, "digital_zero_run", 3, 8),
        (1, "digital_zero_run", 9, 11),
    }
    assert not any(i["right_censored"] for i in intervals)
    assert all(i["byte_span_start"] == 44 + 4 * i["start_frame"] for i in intervals)
    assert result["channels"][0]["longest_observed_zero_run_frames"] == 5


def test_retention_cutoff_does_not_change_sample_counts(tmp_path):
    result, _, intervals = inspect(
        tmp_path,
        [0, 0, 2, 0, 0, 0, 0, 0, 32767],
        qc.DecodePolicy(block_frames=2, zero_run_min_seconds=0.025, endpoint_run_min_frames=2),
    )
    assert result["zero_run_min_frames"] == 3  # ceil, never round downward
    assert [(i["start_frame"], i["stop_frame"]) for i in intervals] == [(3, 8)]
    assert result["channels"][0]["zero_samples"] == 7
    assert result["channels"][0]["positive_endpoint_samples"] == 1


def test_candidate_budget_is_visible_and_no_samples_skipped(tmp_path):
    result, _, intervals = inspect(
        tmp_path,
        [0, 1, 0, 1, 0, 1],
        qc.DecodePolicy(block_frames=2, max_candidate_intervals_per_file=1),
    )
    assert result["decode_complete"] and result["decoded_frames"] == 6
    assert not result["success"] and not result["candidate_intervals_complete"]
    assert result["candidate_intervals_observed"] == 3 and len(intervals) == 1
    assert result["errors"][0]["stage"] == "candidate_retention"


def test_empty_file_is_explicit_and_fractions_are_unknown(tmp_path):
    result, blocks, intervals = inspect(tmp_path, [])
    assert result["success"] and result["empty_data_chunk"]
    assert result["coverage_fraction"] is None
    assert result["channels"][0]["rms"] is None
    assert blocks == intervals == []


def test_report_budget_stops_before_decoding(tmp_path):
    result, blocks, _ = inspect(
        tmp_path, [1] * 20, qc.DecodePolicy(block_frames=2, max_blocks_per_file=2)
    )
    assert not result["success"] and result["decoded_frames"] == 0
    assert blocks == []
    assert result["uninspected_intervals"] == [{"start_frame": 0, "stop_frame": 20}]


@pytest.mark.parametrize(
    "case",
    [
        "truncated",
        "chunk_oversized",
        "frame_alignment",
        "byte_rate",
        "duplicate_data",
        "pcm24",
        "bad_magic",
    ],
)
def test_structure_failures_prevent_false_full_decode(tmp_path, case):
    fmt = bytearray(struct.pack("<HHIIHH", 1, 1, 100, 200, 2, 16))
    chunks = [(b"fmt ", bytes(fmt)), (b"data", b"\x01\x00" * 3)]
    if case == "frame_alignment":
        chunks[-1] = (b"data", b"x")
    if case == "byte_rate":
        struct.pack_into("<I", fmt, 8, 199)
        chunks[0] = (b"fmt ", bytes(fmt))
    if case == "pcm24":
        struct.pack_into("<H", fmt, 14, 24)
        chunks[0] = (b"fmt ", bytes(fmt))
    if case == "duplicate_data":
        chunks.append((b"data", b"\x00\x00"))
    data = riff(chunks)
    if case == "truncated":
        data = data[:-1]
    if case == "chunk_oversized":
        data = data[:40] + struct.pack("<I", 999999) + data[44:]
    if case == "bad_magic":
        data = b"RF64" + data[4:]
    source = tmp_path / "invalid.wav"
    source.write_bytes(data)
    blocks, intervals = [], []
    result = qc.decode_pcm16(source, qc.DecodePolicy(), blocks.append, intervals.append)
    assert not result["success"] and not result["decode_complete"]
    assert blocks == intervals == [] and result["errors"][0]["stage"] == "structure"


def test_unknown_chunk_padding_is_respected(tmp_path):
    source = tmp_path / "odd.wav"
    source.write_bytes(wave_bytes([1, 2, 3], extra=True))
    header = qc.pcm16_header(source)
    assert header.data_offset == 54 and header.frames == 3
    assert header.chunks[0]["pad_bytes"] == 1


@pytest.mark.parametrize("mode", ["error", "short", "extra", "header"])
def test_decoder_disagreement_and_partial_coverage_retained(tmp_path, monkeypatch, mode):
    real = qc.sf.SoundFile

    class InjectedDecoder:
        def __init__(self, path, mode):
            self.inner = real(path, mode)
            self.channels, self.samplerate, self.subtype = (
                self.inner.channels,
                self.inner.samplerate,
                self.inner.subtype,
            )
            self.frames = self.inner.frames + (1 if fault == "header" else 0)
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def read(self, frames, **kwargs):
            self.calls += 1
            if self.calls == 2 and fault == "error":
                raise RuntimeError("synthetic decoding failure")
            if self.calls == 2 and fault == "short":
                return self.inner.read(1, **kwargs)
            if self.calls == 4 and fault == "extra":
                return np.ones((1, 1), dtype=np.int16)
            return self.inner.read(frames, **kwargs)

    fault = mode
    monkeypatch.setattr(qc.sf, "SoundFile", InjectedDecoder)
    result, blocks, intervals = inspect(tmp_path, [0] * 10)
    assert not result["success"] and not result["decode_complete"]
    if mode in {"error", "short"}:
        observed = 4 if mode == "error" else 5
        assert result["decoded_frames"] == observed
        assert result["uninspected_intervals"] == [{"start_frame": observed, "stop_frame": 10}]
        assert result["errors"][0]["start_frame"] == observed
        assert result["errors"][0]["stop_frame"] == 8
        assert intervals[0]["right_censored"]
    elif mode == "extra":
        assert result["coverage_fraction"] == 1 and result["errors"][0]["stage"] == "eof"
    else:
        assert result["decoded_frames"] == 0 and blocks == intervals == []


def make_bundle(tmp_path, files):
    source, baseline = tmp_path / "source", tmp_path / "baseline"
    source.mkdir()
    baseline.mkdir()
    manifest, artifacts = [], []
    for name, data in files.items():
        (source / name).write_bytes(data)
        digest = sha256_file(source / name)
        artifacts.append(
            Artifact(
                artifact_id=stable_id("artifact", [name, None, None, digest]),
                source_path=name,
                sha256=digest,
                size_bytes=len(data),
                lineage="synthetic",
            )
        )
        manifest.append(
            {"source_path": name, "sha256": digest, "before": {"size_bytes": len(data)}}
        )
    write_json(baseline / "source_manifest.json", manifest)
    digest = sha256_file(baseline / "source_manifest.json")
    observations = []
    for art in artifacts:
        raw = {"alias": Path(art.source_path).stem, "canonical_animal_id": None}
        observations.append(
            Observation(
                observation_id=stable_id(
                    "obs", [art.artifact_id, "filename", "recording_alias", raw, PARSER_VERSION]
                ),
                artifact_id=art.artifact_id,
                locator="filename",
                kind="recording_alias",
                raw=raw,
                manifest_sha256=digest,
            )
        )
    bundle = EvidenceBundle(
        manifest_sha256=digest,
        protocol_sha256="a" * 64,
        artifacts=artifacts,
        observations=observations,
    )
    write_json(baseline / "evidence.json", bundle.model_dump(mode="json"))
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(qc.DecodePolicy(block_frames=4).model_dump()))
    cfg = SourceConfig(source_root=source, output_root=tmp_path / "runs")
    return cfg, baseline / "evidence.json", policy


def test_full_cli_provenance_determinism_and_immutability(tmp_path, monkeypatch):
    cfg, baseline, policy = make_bundle(
        tmp_path,
        {"0001.wav": wave_bytes([0, 0, 1, 32767, -32768]), "0011u.wav": wave_bytes([1, 2, 3])},
    )
    before = {p: sha256_file(p) for p in cfg.source_root.iterdir()}
    config = tmp_path / "config.yaml"
    config.write_text("source_root: source\noutput_root: runs\n")
    monkeypatch.chdir(tmp_path.parent)
    first = CliRunner().invoke(
        app,
        [
            "full-decode-pilot",
            "--config",
            str(config),
            "--evidence",
            str(baseline),
            "--policy",
            str(policy),
        ],
    )
    assert first.exit_code == 0, first.output + str(first.exception)
    out = Path(first.stdout.strip())
    second = new_run(cfg, "decode")
    report = qc.full_decode_pilot(cfg, baseline, policy, second)
    assert report["success"] and report["source_verification_passed"]
    assert report["whole_recordings_decoded"] == 2 and report["decoded_frames"] == 8
    assert before == {p: sha256_file(p) for p in before}
    for name in ("recordings.json", "summary.json", "blocks.jsonl", "candidate_intervals.jsonl"):
        assert (out / name).read_bytes() == (second / name).read_bytes()
    assert (out / "run.json").exists() and report["automatic_exclusions"] == 0
    assert report["prior_exposure"][1]["source_path"] == "0011u.wav"
    with pytest.raises(ValueError, match="empty run"):
        qc.full_decode_pilot(cfg, baseline, policy, out)


def test_failed_recording_does_not_disappear_or_block_independent_file(tmp_path):
    cfg, baseline, policy = make_bundle(
        tmp_path, {"bad.wav": b"invalid", "good.wav": wave_bytes([1, 2, 3])}
    )
    out = new_run(cfg, "decode")
    report = qc.full_decode_pilot(cfg, baseline, policy, out)
    assert not report["success"] and report["failed_recordings"] == 1
    assert report["whole_recordings_decoded"] == 1 and report["source_verification_passed"]
    rows = json.loads((out / "recordings.json").read_text())
    assert len(rows) == 2 and rows[0]["decode"]["coverage_fraction"] is None
    assert len(json.loads((out / "failures.json").read_text())) == 1


def test_changed_baseline_and_changed_source_during_decode(tmp_path, monkeypatch):
    cfg, baseline, policy = make_bundle(tmp_path, {"source.wav": wave_bytes([1, 2, 3])})
    decoder = qc.decode_pcm16

    def change_after_decode(path, policy, block_sink, interval_sink):
        result = decoder(path, policy, block_sink, interval_sink)
        value = path.read_bytes()
        path.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
        return result

    monkeypatch.setattr(qc, "decode_pcm16", change_after_decode)
    out = new_run(cfg, "decode")
    report = qc.full_decode_pilot(cfg, baseline, policy, out)
    assert not report["success"] and not report["source_verification_passed"]
    assert report["whole_recordings_decoded"] == 1
    monkeypatch.setattr(qc, "decode_pcm16", decoder)
    second = new_run(cfg, "decode")
    report = qc.full_decode_pilot(cfg, baseline, policy, second)
    assert not report["success"] and report["whole_recordings_decoded"] == 0
    assert json.loads((second / "recordings.json").read_text())[0]["decode"] is None


def test_policy_bounds_and_source_read_budget(tmp_path):
    for changes in (
        {"automatic_exclusions": True},
        {"max_recordings": 13},
        {"zero_run_min_seconds": 0},
        {"block_frames": True},
    ):
        with pytest.raises(ValidationError):
            qc.DecodePolicy.model_validate(changes)
    cfg, baseline, policy = make_bundle(tmp_path, {"source.wav": wave_bytes([1, 2, 3])})
    policy.write_text("max_source_bytes: 1\n")
    with pytest.raises(ValueError, match="read budget"):
        qc.full_decode_pilot(cfg, baseline, policy, new_run(cfg, "decode"))


def test_excluded_output_source_is_never_decoded(tmp_path):
    cfg, baseline, policy = make_bundle(tmp_path, {"source.wav": wave_bytes([1, 2, 3])})
    cfg = SourceConfig(
        source_root=cfg.source_root, output_root=cfg.output_root, exclude_relative=["source.wav"]
    )
    out = new_run(cfg, "decode")
    report = qc.full_decode_pilot(cfg, baseline, policy, out)
    assert not report["success"] and report["hash_bytes_read"] == 0
