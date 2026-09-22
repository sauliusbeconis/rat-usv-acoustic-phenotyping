# M0.2 bounded full-decode QC

This command verifies complete decoding of explicitly selected, previously exposed classic RIFF PCM16 recordings. It supplements M0.1 metadata observations and the earlier sampled audit. It never excludes a recording, assigns biology or changes source bytes.

```bash
uv run --frozen usv full-decode-pilot \
  --config /path/to/config.local.yaml \
  --evidence /path/to/m01-run/evidence.json \
  --policy configs/qc/full_decode.yaml
```

The source/output roots resolve from the config YAML. The evidence bundle and sibling manifest must validate. Only non-archive artifacts with recording-alias observations are selected; other artifacts in that bundle are not decoded. Every selected recording is retained as a result row, including unsupported formats, structural errors and hash mismatches. Excluded subtrees, symlinks and traversal remain disallowed as source inputs.

## Scope and technical reporting policy

This milestone runs on the same four recordings as M0.1. The command has hard bounds of 12 recordings, 512 MiB per file and at most 2 GiB in a configured source pass; the committed default total is 1 GiB. A normal successful execution reads each selected file for a before hash, decodes its audio payload once, then reads the file for an after hash. This does not back up or freeze the collection.

The initial decoder supports format-tag-1 PCM16 with exactly one data chunk and one format chunk. Other encodings, RF64 and ambiguous multiple data/format chunks remain explicit failures for later format-specific work. Unknown chunks are retained in the directory and skipped. No sample rate, amplitude, channel count or source metadata is rewritten.

Block allocation is capped at 1,048,576 samples, with 262,144 frames per block by default. Frame budgets are checked before decoding. The policy permits at most 100,000 block records and retains up to 10,000 candidate intervals per recording by default. Exceeding interval retention does not stop frame decoding or aggregate counting, but the interval list is marked incomplete and the command exits unsuccessfully.

Candidates use these operational definitions:

| Candidate | Definition | Interpretation limit |
| --- | --- | --- |
| Digital-zero run | Consecutive exact zero PCM samples per channel, retained at >= ceil(0.01 × sample rate) frames under the default policy | Possible digital padding/dropout or another source process; not acoustic silence |
| PCM endpoint run | Consecutive samples at either -32768 or +32767, retained from one frame by default | Clipping candidate; saturation or biological invalidity is not established |

All zero and endpoint samples contribute to per-channel counts regardless of interval-retention cutoffs. Runs carry across blocks. Partial decoding marks a run open at the inspected boundary as right-censored. This is a within-file diagnostic; it does not infer what happened outside a recording.

These cutoffs are technical reporting choices, not biological thresholds, validated QC criteria or exclusion rules. Full decoding does not assess call presence, acoustic noise, annotation accuracy or missed detections. The systematic annotation pilot and targeted anomaly review remain distinct future steps.

## Verification and output

Before decoding, independently validate RIFF length against actual file size; all chunk bounds/padding; PCM16 sample format, block alignment and byte rate; and complete data-frame packing. Compare these with the decoder's sample rate, channels and frame count. Sequential reads request integer PCM16 samples using [SoundFile's read interface](https://python-soundfile.readthedocs.io/en/latest/#soundfile.SoundFile.read). Verify EOF after the expected frame count.

Per-channel sample counts, sums and squared sums use integer arithmetic; the final mean and RMS are derived in signed PCM16 sample units. No physical sound-pressure calibration is implied. Byte spans for multichannel intervals cover complete interleaved frames and are labelled accordingly. Frame indexes are zero-based and intervals are half-open.

| File | Contents |
| --- | --- |
| recordings.json | Every selected source, validated chunk directory, header, per-channel counts, exact inspected/uninspected frame ranges, decoder errors, before/after signatures and hashes |
| blocks.jsonl | Every processed block, keyed by source artifact ID, with frame ranges and separate channel statistics |
| candidate_intervals.jsonl | Retained runs, channel, frames, seconds, source-byte span and right-censoring status |
| failures.json | Retained unsuccessful recordings, including partial results; zero-based error ranges localize the requested unread block, not an exact corrupt sample |
| summary.json | Coverage denominators, source/evidence/protocol/policy digests, failures, additional prior exposure and scope limits |
| selection.json, policy.json | Exact selected artifacts and resolved reporting policy |
| run.json, code_snapshot.json | Code/config/lock/input/output hashes, package and decoder versions, Git state and reproducibility snapshot |

A frame-coverage fraction of one alone does not certify success: the EOF, candidate-retention and source-integrity checks also matter. An empty data chunk has an explicit empty flag and null coverage fraction, mean and RMS. It does not establish absence of vocalizations. A source that changed can retain diagnostic results but cannot receive a successful verified result.

Synthetic tests cover mono/stereo integer statistics, exact endpoints, channel separation, zero-run retention cutoffs, continuity across block sizes, empty files, malformed/truncated RIFF, frame packing, unsupported formats, decoder/header disagreements, short reads, EOF errors, partial coverage, candidate/report budgets, hash mismatches, source changes, output separation, CLI failures and deterministic output. Run the five AGENTS.md checks and the isolated candidate-checkout check before delivery. Private results stay under new ignored runs; the M0.2 handoff index is `.local/setup/m02_run_index.json` after acceptance.
