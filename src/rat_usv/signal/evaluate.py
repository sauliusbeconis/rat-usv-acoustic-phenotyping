"""Match detected calls to reference calls by temporal overlap (one-to-one)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class Interval(Protocol):
    @property
    def start_s(self) -> float: ...
    @property
    def end_s(self) -> float: ...


@dataclass(frozen=True)
class MatchResult:
    true_positive: int
    false_positive: int
    false_negative: int
    pairs: list[tuple[int, int]]

    @property
    def precision(self) -> float | None:
        n = self.true_positive + self.false_positive
        return self.true_positive / n if n else None

    @property
    def recall(self) -> float | None:
        n = self.true_positive + self.false_negative
        return self.true_positive / n if n else None


def match_calls(
    detected: Sequence[Interval], reference: Sequence[Interval], min_overlap_s: float = 0.0
) -> MatchResult:
    """Greedy one-to-one matching on largest overlap; a pair needs overlap > min_overlap_s."""
    candidates = []
    for i, d in enumerate(detected):
        for j, r in enumerate(reference):
            overlap = min(d.end_s, r.end_s) - max(d.start_s, r.start_s)
            if overlap > min_overlap_s:
                candidates.append((overlap, i, j))
    used_d: set[int] = set()
    used_r: set[int] = set()
    pairs = []
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
            pairs.append((i, j))
    tp = len(pairs)
    return MatchResult(tp, len(detected) - tp, len(reference) - tp, sorted(pairs))
