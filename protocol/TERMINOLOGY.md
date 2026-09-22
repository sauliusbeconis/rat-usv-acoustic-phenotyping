# Terminology and uncertainty

Animal: biological individual; primary biological unit, subject to litter-level allocation. Call: acoustic event under a validated detector/annotation definition. Bout: operational temporal grouping with a justified inter-call threshold. Session: acquisition event; may include multiple files. Animal-day: repeated biological observation with a confirmed day definition. Litter/dam: relatedness and maternal hierarchy, not inferred from filename resemblance.

Postnatal day: integer day relative to an evidenced birth reference; P0/P1 conventions must be reconciled. Recording date is not necessarily postnatal day. Alias: exact historical string (leading zeros/case retained) within a namespace. Canonical ID: stable reviewed identity, never recycled. Recording ID identifies an observation; SHA-256 identifies bytes, not an animal.

Mapping statuses: `confirmed` (reviewed assignment), `probable` (favored candidate with uncertainty), `ambiguous` (multiple plausible choices), `conflicting` (contradictory evidence), `unresolved` (no justified assignment). Probable is not permitted as an invisible substitute for confirmed.

Evidence classes: EXACT = explicit source linkage with precise locator; STRONG = independently corroborated linkage; MODERATE = partial corroboration with missing detail; WEAK = naming/temporal resemblance only; CONFLICT = incompatible evidence; UNRESOLVED = insufficient evidence. Filename seeding starts UNRESOLVED. Evidence strength alone does not automatically confirm an identity. Keep source statements separately from the interpretation.

Developmental acoustic state: operational model representation; not established neural state or causally identified biology. NASS: Neuroacoustic State-Space, internal working concept. Explainability: testable relation between a model output and observed acoustic properties/real calls; attention or attractive plots alone do not establish explanation.
