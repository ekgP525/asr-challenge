"""Track A: character error rate (CER).

Operates on `normalizer.normalize_nospace()` output. CER is corpus-level:
sum of edit operations / sum of reference characters, in percent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .normalizer import normalize_nospace


def levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Edit distance (insert/delete/substitute, unit cost)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


@dataclass
class CERAccumulator:
    edits: int = 0
    ref_chars: int = 0
    n_utts: int = 0

    def add(self, ref: str, hyp: str) -> int:
        r = normalize_nospace(ref)
        h = normalize_nospace(hyp)
        d = levenshtein(r, h)
        self.edits += d
        self.ref_chars += len(r)
        self.n_utts += 1
        return d

    @property
    def cer(self) -> float:
        return 100.0 * self.edits / self.ref_chars if self.ref_chars else 0.0


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    acc = CERAccumulator()
    for r, h in zip(refs, hyps):
        acc.add(r, h)
    return acc.cer


def utterance_cer(ref: str, hyp: str) -> float:
    r = normalize_nospace(ref)
    return 100.0 * levenshtein(r, normalize_nospace(hyp)) / len(r) if r else 0.0
