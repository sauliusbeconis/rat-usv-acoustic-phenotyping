# ADR-0001: preserve sources and implement M0 infrastructure first

Status: accepted for infrastructure by the explicit setup request, 2026-09-21.

The source collection predates this repository and contains recordings, repeated historical analysis trees, workbooks and documents. Preserve all source paths and bytes. Clone the existing GitHub repository below the collection and explicitly exclude it from traversal. Store private output in ignored .local directories; keep registry biological tables empty until evidence reconstruction.

Inventory and WAV metadata/sample-window audit are implemented. Identity-resolution automation, full Excel/R interpretation, final splits and training are deferred to M0 planning and review. Bounded hash coverage is reported instead of suggesting a full source freeze. No source artifact is relocated or renamed. No remote push is performed.

Decision consequence: infrastructure tests can pass while biological readiness remains false. Existing copies and ambiguous filename semantics require explicit follow-up. Formal scientific protocol acceptance is still pending.
