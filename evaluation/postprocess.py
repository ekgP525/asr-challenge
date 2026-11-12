"""Output guards a participant system may apply to its final text (deterministic, negligible cost).

collapse_repeats(text): a streaming LLM recognizer can fall into a loop on noisy or truncated audio
(S2 at 0 dB produced 48x the reference length by repeating one clause). When a substring of
`min_len`+ characters repeats back-to-back `min_repeats`+ times, keep one copy and drop everything
from the second copy on; the text after a loop is never trustworthy. Normal speech does not repeat a
20-character clause three times in a row, so the guard is inert on ordinary output.
"""
from __future__ import annotations


def collapse_repeats(text: str, min_len: int = 6, min_repeats: int = 3, max_len: int = 60) -> str:
    s = text or ""
    n = len(s)
    for L in range(min_len, min(max_len, n // min_repeats) + 1):
        i = 0
        while i + L * min_repeats <= n:
            unit = s[i: i + L]
            if unit.strip() and s[i + L: i + 2 * L] == unit and s[i + 2 * L: i + 3 * L] == unit:
                return s[: i + L].rstrip()
            i += 1
    return s
