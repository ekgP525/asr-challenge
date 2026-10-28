"""Track B metrics computed from `stream_sim.UttRecord` dicts.

Per condition:
  p50 / p90            final latency = t_final - speech_end (s)
  endpoint             median of (vad_end - speech_end) (s); vad_end = audio end when VAD never fired
  early_cut            % of utterances with vad_end < speech_end
  revision             % of partial updates that rewrote already-shown text (streaming only)
  first_partial        median of (t_first_partial - speech_start) (s), streaming only
  rtfx                 sum(audio_duration) / sum(processing_time)

Revision rule (character-level prefix test): an update from partial A to partial B
is a revision when A (whitespace-collapsed) is not a prefix of B. The first
partial is never a revision; the final text counts as the last update.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable, List, Optional

_WS = re.compile(r"\s+")


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def _collapse(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def revision_count(partials: Iterable[dict]):
    """(revisions, updates) over a partial-result sequence."""
    texts = [_collapse(p["text"]) for p in partials]
    rev = upd = 0
    for a, b in zip(texts, texts[1:]):
        upd += 1
        if not b.startswith(a):
            rev += 1
    return rev, upd


def final_latency(r: dict) -> float:
    return r["t_final"] - r["speech_end"]


def endpoint_error(r: dict) -> float:
    end = r["vad_end"] if r.get("vad_end") is not None else r["audio_duration"]
    return end - r["speech_end"]


def summarize(records: List[dict], streaming: Optional[bool] = None) -> dict:
    """Aggregate one condition. `streaming=None` -> inferred from the records."""
    if not records:
        return {"n": 0}
    if streaming is None:
        streaming = any(len(r.get("partials", [])) > 1 or r.get("t_first_partial") is not None
                        for r in records)
    finals = [final_latency(r) for r in records]
    ends = [endpoint_error(r) for r in records]
    early = sum(1 for e in ends if e < 0)
    audio = sum(r["audio_duration"] for r in records)
    proc = sum(r["processing_time"] for r in records)
    out = {
        "n": len(records),
        "p50": round(percentile(finals, 50), 4),
        "p90": round(percentile(finals, 90), 4),
        "endpoint": round(statistics.median(ends), 4),
        "early_cut": round(100.0 * early / len(records), 2),
        "revision": None,
        "first_partial": None,
        "rtfx": round(audio / proc, 2) if proc > 0 else None,
    }
    if streaming:
        rev = upd = 0
        for r in records:
            a, b = revision_count(r.get("partials", []))
            rev += a
            upd += b
        out["revision"] = round(100.0 * rev / upd, 2) if upd else 0.0
        firsts = [r["t_first_partial"] - r["speech_start"] for r in records
                  if r.get("t_first_partial") is not None]
        out["first_partial"] = round(statistics.median(firsts), 4) if firsts else None
    return out
