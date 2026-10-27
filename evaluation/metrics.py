"""Accuracy metrics: character error rate (CER) and Term Recall.

Both operate on `normalizer.normalize_nospace()` output.

CER is corpus-level: sum of edit operations / sum of reference characters,
reported in percent. Term Recall is corpus-level: hit terms / total terms in
the references, in percent. A term is a hit when it appears as a substring of
the normalized, space-free hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

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


@dataclass
class TermRecallAccumulator:
    hits: int = 0
    total: int = 0
    missed: List[dict] = field(default_factory=list)

    def add(self, utt_id: str, hyp: str, terms: Iterable[str]) -> int:
        h = normalize_nospace(hyp)
        n_hit = 0
        for t in terms:
            tn = normalize_nospace(t)
            if not tn:
                continue
            self.total += 1
            if tn in h:
                n_hit += 1
            else:
                self.missed.append({"id": utt_id, "term": t})
        self.hits += n_hit
        return n_hit

    @property
    def term_recall(self) -> float:
        return 100.0 * self.hits / self.total if self.total else 0.0


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    acc = CERAccumulator()
    for r, h in zip(refs, hyps):
        acc.add(r, h)
    return acc.cer


def term_recall(hyps: Sequence[str], terms_per_utt: Sequence[Sequence[str]]) -> float:
    acc = TermRecallAccumulator()
    for i, (h, terms) in enumerate(zip(hyps, terms_per_utt)):
        acc.add(str(i), h, terms)
    return acc.term_recall


def load_terms(path: str) -> List[str]:
    """Load data/terms.txt (one term per line, '#' comments, blank lines ignored)."""
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return terms


def find_terms(text: str, terms: Iterable[str]) -> List[str]:
    """Terms from the dictionary present in `text` (used to fill manifest 'terms')."""
    t = normalize_nospace(text)
    return [term for term in terms if normalize_nospace(term) and normalize_nospace(term) in t]
