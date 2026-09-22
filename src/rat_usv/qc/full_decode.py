"""Bounded whole-recording PCM16 decoding. Digital diagnostics never exclude sources."""

import json
import math
import struct
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from rat_usv.config import HashPolicy, SourceConfig
from rat_usv.evidence.pilot import signature, validate_bundle_file
from rat_usv.provenance.io import Hasher, record_run, safe_source, sha256_file, write_json

DECODER_VERSION = "full-decode/1"
Sink = Callable[[dict[str, Any]], None]


class DecodePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    block_frames: int = Field(default=262144, ge=1, le=1048576)
    max_recordings: int = Field(default=12, ge=1, le=12)
    max_source_bytes: int = Field(default=1073741824, ge=1, le=2147483648)
    max_file_bytes: int = Field(default=536870912, ge=1, le=536870912)
    max_blocks_per_file: int = Field(default=100000, ge=1, le=100000)
    max_candidate_intervals_per_file: int = Field(default=10000, ge=1, le=100000)
    zero_run_min_seconds: float = Field(default=0.01, gt=0, le=10)
    endpoint_run_min_frames: int = Field(default=1, ge=1, le=1048576)
    automatic_exclusions: Literal[False] = False


@dataclass(frozen=True)
class Pcm16Header:
    sample_rate_hz: int
    channels: int
    frames: int
    block_align: int
    data_offset: int
    data_bytes: int
    chunks: list[dict[str, Any]]


def pcm16_header(path: Path) -> Pcm16Header:
    """Validate actual RIFF extents before trusting a decoder's frame count."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("Unsupported/truncated RIFF/WAVE header")
        if struct.unpack_from("<I", header, 4)[0] + 8 != size:
            raise ValueError("RIFF declared length differs from actual file length")
        offset = 12
        chunks: list[dict[str, Any]] = []
        formats: list[bytes] = []
        payloads: list[tuple[int, int]] = []
        while offset < size:
            if len(chunks) >= 10000:
                raise ValueError("RIFF chunk-directory budget exceeded")
            stream.seek(offset)
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated chunk header at byte {offset}")
            tag, length = struct.unpack("<4sI", raw)
            end = offset + 8 + length + length % 2
            if end > size:
                raise ValueError(f"Chunk exceeds RIFF boundary at byte {offset}")
            chunks.append(
                {
                    "tag_hex": tag.hex(),
                    "offset": offset,
                    "payload_offset": offset + 8,
                    "size_bytes": length,
                    "pad_bytes": length % 2,
                }
            )
            if tag == b"fmt ":
                if length < 16 or length > 65536:
                    raise ValueError("Invalid or oversized fmt chunk")
                formats.append(stream.read(length))
            elif tag == b"data":
                payloads.append((offset + 8, length))
            offset = end
    if len(formats) != 1 or len(payloads) != 1:
        raise ValueError("PCM16 pilot requires exactly one fmt and one data chunk")
    encoding, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", formats[0])
    if encoding != 1 or bits != 16:
        raise ValueError("Unsupported encoding: pilot accepts classic RIFF PCM16 only")
    if not 1 <= channels <= 64 or not 1 <= rate <= 1000000:
        raise ValueError("Channel count or sample rate outside pilot bounds")
    if alignment != channels * 2 or byte_rate != rate * alignment:
        raise ValueError("Inconsistent PCM16 block alignment or byte rate")
    data_offset, data_bytes = payloads[0]
    if data_bytes % alignment:
        raise ValueError("Data chunk ends inside a PCM frame")
    return Pcm16Header(
        rate, channels, data_bytes // alignment, alignment, data_offset, data_bytes, chunks
    )


@dataclass
class ChannelStats:
    samples: int = 0
    total: int = 0
    sum_squares: int = 0
    minimum: int | None = None
    maximum: int | None = None
    zero_samples: int = 0
    positive_endpoint_samples: int = 0
    negative_endpoint_samples: int = 0

    def add(self, values: NDArray[np.int16]) -> dict[str, Any]:
        integers = values.astype(np.int64)
        stats = {
            "samples": len(values),
            "sum": int(integers.sum()),
            "sum_squares": int(np.dot(integers, integers)),
            "minimum": int(values.min()),
            "maximum": int(values.max()),
            "digital_zero_samples": int(np.count_nonzero(values == 0)),
            "positive_endpoint_samples": int(np.count_nonzero(values == 32767)),
            "negative_endpoint_samples": int(np.count_nonzero(values == -32768)),
        }
        self.samples += stats["samples"]
        self.total += stats["sum"]
        self.sum_squares += stats["sum_squares"]
        self.minimum = (
            stats["minimum"] if self.minimum is None else min(self.minimum, stats["minimum"])
        )
        self.maximum = (
            stats["maximum"] if self.maximum is None else max(self.maximum, stats["maximum"])
        )
        self.zero_samples += stats["digital_zero_samples"]
        self.positive_endpoint_samples += stats["positive_endpoint_samples"]
        self.negative_endpoint_samples += stats["negative_endpoint_samples"]
        return stats

    def report(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "amplitude_units": "signed PCM16 sample values",
            "mean_dc": self.total / self.samples if self.samples else None,
            "rms": math.sqrt(self.sum_squares / self.samples) if self.samples else None,
            "digital_zero_fraction": self.zero_samples / self.samples if self.samples else None,
            "endpoint_fraction": (self.positive_endpoint_samples + self.negative_endpoint_samples)
            / self.samples
            if self.samples
            else None,
        }


class RunTracker:
    """Carry contiguous runs across block boundaries, including partial-decode edges."""

    def __init__(self, kind: str, channel: int, minimum_frames: int, sink: Sink) -> None:
        self.kind, self.channel, self.minimum_frames, self.sink = (
            kind,
            channel,
            minimum_frames,
            sink,
        )
        self.start: int | None = None
        self.longest_observed_frames = 0

    def close(self, stop: int, right_censored: bool = False) -> None:
        if self.start is None:
            return
        length = stop - self.start
        self.longest_observed_frames = max(self.longest_observed_frames, length)
        if length >= self.minimum_frames:
            self.sink(
                {
                    "kind": self.kind,
                    "channel_index": self.channel,
                    "start_frame": self.start,
                    "stop_frame": stop,
                    "frames": length,
                    "right_censored": right_censored,
                }
            )
        self.start = None

    def add(self, mask: NDArray[np.bool_], offset: int) -> None:
        if not len(mask):
            return
        if self.start is not None and not mask[0]:
            self.close(offset)
        padded = np.concatenate((np.array([False]), mask, np.array([False])))
        boundaries = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in zip(boundaries[::2], boundaries[1::2], strict=True):
            if self.start is None:
                self.start = offset + int(start)
            if stop < len(mask):
                self.close(offset + int(stop))


def decode_pcm16(
    path: Path, policy: DecodePolicy, block_sink: Sink, interval_sink: Sink
) -> dict[str, Any]:
    """Sequential integer decoding with explicit coverage and retained partial failures."""
    report: dict[str, Any] = {
        "header": None,
        "decoded_frames": 0,
        "coverage_fraction": None,
        "coverage_intervals": [],
        "uninspected_intervals": [],
        "decode_complete": False,
        "candidate_intervals_complete": True,
        "candidate_intervals_observed": 0,
        "candidate_intervals_retained": 0,
        "errors": [],
        "channels": [],
        "automatic_exclusion": False,
    }
    stats: list[ChannelStats] = []
    trackers: list[RunTracker] = []
    header: Pcm16Header | None = None
    start, requested_stop, blocks = 0, 0, 0
    stage = "structure"
    try:
        header = pcm16_header(path)
        report["header"] = asdict(header)
        if policy.block_frames * header.channels > 1048576:
            raise ValueError("Decoded block sample budget exceeded")
        if math.ceil(header.frames / policy.block_frames) > policy.max_blocks_per_file:
            raise ValueError("Block-report budget exceeded before decoding")
        report["zero_run_min_frames"] = max(
            1, math.ceil(policy.zero_run_min_seconds * header.sample_rate_hz)
        )

        def emit_interval(item: dict[str, Any]) -> None:
            assert header is not None
            report["candidate_intervals_observed"] += 1
            if report["candidate_intervals_retained"] >= policy.max_candidate_intervals_per_file:
                report["candidate_intervals_complete"] = False
                return
            interval_sink(
                {
                    **item,
                    "start_seconds": item["start_frame"] / header.sample_rate_hz,
                    "stop_seconds": item["stop_frame"] / header.sample_rate_hz,
                    "byte_span_start": header.data_offset
                    + item["start_frame"] * header.block_align,
                    "byte_span_stop": header.data_offset + item["stop_frame"] * header.block_align,
                    "byte_span_includes_all_channels": True,
                }
            )
            report["candidate_intervals_retained"] += 1

        stats = [ChannelStats() for _ in range(header.channels)]
        for channel in range(header.channels):
            trackers.extend(
                [
                    RunTracker(
                        "digital_zero_run", channel, report["zero_run_min_frames"], emit_interval
                    ),
                    RunTracker(
                        "pcm_endpoint_run", channel, policy.endpoint_run_min_frames, emit_interval
                    ),
                ]
            )
        stage = "decoder_header"
        with sf.SoundFile(path, mode="r") as audio:
            if (audio.samplerate, audio.channels, audio.frames, audio.subtype) != (
                header.sample_rate_hz,
                header.channels,
                header.frames,
                "PCM_16",
            ):
                raise ValueError("Decoder header disagrees with validated PCM16 structure")
            stage = "decode"
            while start < header.frames:
                requested_stop = min(start + policy.block_frames, header.frames)
                values = audio.read(requested_stop - start, dtype="int16", always_2d=True)
                if (
                    values.dtype != np.int16
                    or values.ndim != 2
                    or values.shape[1] != header.channels
                    or len(values) > requested_stop - start
                ):
                    raise ValueError("Unexpected decoder block shape or dtype")
                if len(values):
                    channel_rows = []
                    for channel in range(header.channels):
                        x = values[:, channel]
                        channel_rows.append({"channel_index": channel, **stats[channel].add(x)})
                        trackers[2 * channel].add(x == 0, start)
                        trackers[2 * channel + 1].add((x == -32768) | (x == 32767), start)
                    stop = start + len(values)
                    block_sink({"start_frame": start, "stop_frame": stop, "channels": channel_rows})
                    start = stop
                    blocks += 1
                if start != requested_stop:
                    raise ValueError("Short decoder read; missing samples are not filled")
            stage = "eof"
            if len(audio.read(1, dtype="int16", always_2d=True)):
                raise ValueError("Decoder returned frames beyond declared data chunk")
            report["decode_complete"] = True
    except (OSError, RuntimeError, ValueError) as error:
        report["errors"].append(
            {
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "start_frame": start if header is not None else None,
                "stop_frame": requested_stop if stage == "decode" else None,
                "localization": "Requested unread block, not identification of a corrupt sample"
                if stage == "decode"
                else "File/structure/EOF check",
            }
        )
    for tracker in trackers:
        tracker.close(start, right_censored=not report["decode_complete"])
    report["decoded_frames"], report["decoded_blocks"] = start, blocks
    report["channels"] = [
        {
            "channel_index": i,
            **stat.report(),
            "longest_observed_zero_run_frames": trackers[2 * i].longest_observed_frames,
            "longest_observed_endpoint_run_frames": trackers[2 * i + 1].longest_observed_frames,
        }
        for i, stat in enumerate(stats)
    ]
    if start:
        report["coverage_intervals"] = [{"start_frame": 0, "stop_frame": start}]
    if header is not None:
        report["coverage_fraction"] = start / header.frames if header.frames else None
        report["empty_data_chunk"] = header.frames == 0
        if start < header.frames:
            report["uninspected_intervals"] = [{"start_frame": start, "stop_frame": header.frames}]
    if not report["candidate_intervals_complete"]:
        report["errors"].append(
            {
                "stage": "candidate_retention",
                "error": "Candidate interval budget exceeded; counts retained but interval list is incomplete",
            }
        )
    report["success"] = report["decode_complete"] and not report["errors"]
    return report


def full_decode_pilot(
    config: SourceConfig, evidence_path: Path, policy_path: Path, out: Path
) -> dict[str, Any]:
    """Decode only the previously exposed WAVs explicitly present in an evidence bundle."""
    if not out.resolve().is_relative_to(config.output_root) or any(out.iterdir()):
        raise ValueError("Full decode output must be an empty run under output_root")
    policy = DecodePolicy.model_validate(yaml.safe_load(policy_path.read_text()))
    evidence = validate_bundle_file(evidence_path)
    ids = {o.artifact_id for o in evidence.observations if o.kind == "recording_alias"}
    selected = [a for a in evidence.artifacts if a.artifact_id in ids and a.member_name is None]
    if not selected or len(selected) > policy.max_recordings:
        raise ValueError("Selected recording count outside explicit pilot bounds")
    if len({a.source_path for a in selected}) != len(selected):
        raise ValueError("Duplicate recording selection; no implicit deduplication")
    if sum(a.size_bytes for a in selected) > policy.max_source_bytes or any(
        a.size_bytes > policy.max_file_bytes for a in selected
    ):
        raise ValueError("Selected sources exceed declared read budget")
    write_json(out / "policy.json", policy.model_dump(mode="json"))
    write_json(out / "selection.json", [a.model_dump(mode="json") for a in selected])
    rows = []
    hash_policy = HashPolicy(
        mode="budget", max_file_bytes=policy.max_file_bytes, max_total_bytes=policy.max_source_bytes
    )
    before_hasher, after_hasher = Hasher(hash_policy), Hasher(hash_policy)
    with (
        (out / "blocks.jsonl").open("x") as block_file,
        (out / "candidate_intervals.jsonl").open("x") as interval_file,
    ):
        for artifact in selected:
            row: dict[str, Any] = {
                "artifact_id": artifact.artifact_id,
                "source_path": artifact.source_path,
                "expected_sha256": artifact.sha256,
                "before_sha256": None,
                "after_sha256": None,
                "source_unchanged": False,
                "decode": None,
                "errors": [],
                "success": False,
            }
            path: Path | None = None
            before: dict[str, int] | None = None
            try:
                path = safe_source(config.source_root, artifact.source_path)
                if any(
                    path.relative_to(config.source_root).is_relative_to(x)
                    for x in config.exclude_relative
                ):
                    path = None
                    raise ValueError("Selected recording is inside an excluded/output subtree")
                before = signature(path)
                row["before_signature"] = before
                if (
                    before["size_bytes"] != artifact.size_bytes
                    or before["size_bytes"] > policy.max_file_bytes
                ):
                    raise ValueError("Selected source size differs from evidence or exceeds budget")
                digest, state = before_hasher.compute(path)
                row["before_sha256"] = digest
                row["before_hash_status"] = state
                if state != "hashed" or digest != artifact.sha256:
                    raise ValueError("Source hash differs from evidence; decoding stopped")

                def block_sink(
                    item: dict[str, Any], artifact_id: str = artifact.artifact_id
                ) -> None:
                    block_file.write(
                        json.dumps({"artifact_id": artifact_id, **item}, allow_nan=False) + "\n"
                    )

                def interval_sink(
                    item: dict[str, Any], artifact_id: str = artifact.artifact_id
                ) -> None:
                    interval_file.write(
                        json.dumps({"artifact_id": artifact_id, **item}, allow_nan=False) + "\n"
                    )

                row["decode"] = decode_pcm16(path, policy, block_sink, interval_sink)
            except (OSError, RuntimeError, ValueError) as error:
                row["errors"].append(f"{type(error).__name__}: {error}")
            finally:
                if path is not None and before is not None and row["before_sha256"] is not None:
                    try:
                        if path.stat().st_size > policy.max_file_bytes:
                            raise ValueError("Source grew beyond hash budget")
                        digest, state = after_hasher.compute(path)
                        row["after_hash_status"] = state
                        after = signature(path)
                        row["after_sha256"], row["after_signature"] = digest, after
                        row["source_unchanged"] = (
                            state == "hashed"
                            and digest == row["before_sha256"] == artifact.sha256
                            and before == after
                        )
                        if not row["source_unchanged"]:
                            raise ValueError("Source changed or failed evidence-hash verification")
                    except (OSError, ValueError) as error:
                        row["errors"].append(f"Post-read verification: {error}")
            row["success"] = bool(
                row["source_unchanged"]
                and row["decode"]
                and row["decode"]["success"]
                and not row["errors"]
            )
            rows.append(row)
    write_json(out / "recordings.json", rows)
    failures = [r for r in rows if not r["success"]]
    write_json(out / "failures.json", failures)
    report = {
        "schema_version": 1,
        "decoder_version": DECODER_VERSION,
        "evidence_sha256": sha256_file(evidence_path),
        "source_manifest_sha256": evidence.manifest_sha256,
        "protocol_sha256": evidence.protocol_sha256,
        "policy_sha256": sha256_file(policy_path),
        "selected_recordings": len(selected),
        "selected_bytes": sum(a.size_bytes for a in selected),
        "hash_bytes_read": before_hasher.bytes_read + after_hasher.bytes_read,
        "whole_recordings_decoded": sum(
            bool(r["decode"] and r["decode"]["decode_complete"]) for r in rows
        ),
        "decoded_frames": sum(r["decode"]["decoded_frames"] for r in rows if r["decode"]),
        "source_verification_passed": all(r["source_unchanged"] for r in rows),
        "success": not failures,
        "failed_recordings": len(failures),
        "candidate_intervals_retained": sum(
            r["decode"]["candidate_intervals_retained"] for r in rows if r["decode"]
        ),
        "errors_by_stage": dict(
            Counter(e["stage"] for r in rows if r["decode"] for e in r["decode"]["errors"])
        ),
        "prior_exposure": [
            {
                "source_path": a.source_path,
                "kind": "full waveform decoded for technical QC",
                "canonical_animal_id": None,
            }
            for a in selected
        ],
        "canonical_assignments": 0,
        "automatic_exclusions": 0,
        "interpretation": "Exact digital zeros are not acoustic silence. Endpoint and zero runs are review candidates, not confirmed clipping/dropout. Full decode does not assess call presence, noise, detector recall or future test eligibility.",
        "collection_freeze": False,
    }
    write_json(out / "summary.json", report)
    (out / "review.md").write_text(
        "# M0.2 full-decode QC pilot\n\n"
        f"Status: {'completed; scientific review pending' if report['success'] else 'incomplete; inspect failures.json'}. "
        f"{report['whole_recordings_decoded']}/{len(selected)} recordings fully decoded. Source verification: {report['source_verification_passed']}.\n\n"
        "recordings.json retains validated chunk structure, per-channel counts, exact frame coverage, uninspected/error intervals and before/after hashes. blocks.jsonl contains every processed block; candidate_intervals.jsonl uses zero-based, half-open frame ranges.\n\n"
        "The policy's candidate-retention cutoffs are technical reporting choices, not biological or exclusion thresholds. All-source freeze, human annotation, identities, protocol lock, splits and ML remain pending. No source or canonical registry is modified.\n"
    )
    record_run(
        out,
        "full-decode-pilot",
        config,
        [evidence_path, evidence_path.parent / "source_manifest.json", policy_path],
    )
    return report
