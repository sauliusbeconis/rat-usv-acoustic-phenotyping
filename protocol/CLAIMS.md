# Claim ledger and evidence states

States describe the scope of evidence; they are not a universal linear ranking.

| State | Meaning |
|---|---|
| HYPOTHESIS | Testable proposal; no empirical support asserted |
| OBSERVED | Descriptive finding in a specified dataset/run; no independent validation |
| REPLICATED | Specified finding reproduced in an independent run/sample; disclose what is independent |
| INTERNALLY_VALIDATED | Predefined evaluation with protected biological holdouts inside the source study |
| EXTERNALLY_VALIDATED | Locked pipeline evaluated in an independent appropriate cohort/domain |
| SUPPORTED | Narrow claim meets its predeclared support criterion with uncertainty and scope explicitly reviewed |
| REJECTED | Predeclared criterion contradicts the claim; retain evidence and alternatives |

No automatic state promotion. SUPPORTED does not imply causality or external validity; REJECTED is preserved. A failed or incomplete run does not by itself reject a biological hypothesis.

| Claim ID | Claim | State | Data/run/figure | Limits / next evidence |
|---|---|---|---|---|
| H-001 | Multidimensional USV structure may contain developmental information beyond conventional acoustic summaries | HYPOTHESIS | None | Endpoint and cohort to be defined after M0; compare against strong classical baselines |
| H-002 | Temporal organization may add information beyond pooled call features | HYPOTHESIS | None | Sequence-shuffle and pooled baselines, controlled animal/litter holdouts |
| H-003 | Explicit morphology may improve robustness or explanation over spectral features alone | HYPOTHESIS | None | Branch ablations, detector uncertainty, real-call validation |

For each empirical update record: claim wording/version, estimand, protocol hash, source/manifest/split hashes, run IDs, code/config, independent N, effect and interval, multiplicity, failed controls, figure IDs, reviewer/date and prior state. Technical file counts belong in setup/atlas reports and are not biological claims.
