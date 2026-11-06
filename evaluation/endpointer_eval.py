"""Evaluate an endpointer alone (no ASR, no GPU): endpoint delay, early cuts, misses per condition.

    python evaluation/endpointer_eval.py --spec default
    python evaluation/endpointer_eval.py --spec '{"type":"silero","vad":{"min_silence_duration_ms":350}}'
    python evaluation/endpointer_eval.py --spec participants/S0_energy_example.json   # reads its "endpointer" key
    python evaluation/endpointer_eval.py --spec default --conditions clean,snr0 --limit 200 --out runs/ep_default.jsonl

Runs the real simulator (evaluation/stream_sim.run_streaming) with a system that returns no text, so the
endpointer sees exactly what it would see in a measured run and its wall-clock cost is charged the same way.
Reported per condition (seconds unless %): endpoint delay = vad_end - speech_end for utterances detected
after the true end (p50 / p90 / max), early_cut = % with vad_end < speech_end, missed = % never started,
start_err = median (vad_start - speech_start), and the latency budget left for the ASR under the
1.0 s constraint: budget = limit - endpoint_delay_p90.

Why this matters: the final latency of a streaming system is endpoint delay + a small tail compute
(B4: p90 0.15 s). Whatever endpointer you build, its p90 end delay must leave room for that tail.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.latency import percentile  # noqa: E402
from evaluation.stream_sim import build_endpointer, describe_endpointer, load_config, run_streaming  # noqa: E402

CONDITIONS = ["clean", "snr20", "snr10", "snr5", "snr0"]


class Silent:
    """Streaming system that never produces text (endpointer-only measurement)."""
    text = ""
    def reset(self): ...
    def push(self, chunk, sr): return None
    def finalize(self): return ""


class Replay(Silent):
    """Replays partial transcripts recorded in a latency.jsonl so a text-aware endpointer sees what the
    ASR had produced by each chunk's arrival. Exact when the endpointer's start detection equals the
    baseline's (same gate opening -> same pushes); the compute delay of the recorded run is ignored."""
    def __init__(self, path, sr):
        self.sr = sr
        self.partials = {}
        for r in read_jsonl(path):
            self.partials[r["id"]] = [(p["t"], p["text"]) for p in r["partials"]]
        self.cur = []
        self.fed = 0
        self.text = ""
    def load(self, uid):
        self.cur = self.partials.get(uid, [])
    def reset(self):
        self.fed = 0
        self.text = ""
    def push(self, chunk, sr):
        self.fed += len(chunk)
        # a partial recorded at t was produced by the push of the chunk that arrived at floor(t) (its
        # timestamp includes that push's compute), so it is visible to the next feed(); allow that slack
        t = self.fed / self.sr + 0.45
        avail = [x for x in self.cur if x[0] <= t]
        self.text = avail[-1][1] if avail else ""
        return None


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_audio(path, sr):
    import numpy as np
    import soundfile as sf
    a, fsr = sf.read(path, dtype="float32")
    assert fsr == sr, (path, fsr)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return np.ascontiguousarray(a)


def parse_spec(s: str):
    if s == "default":
        return None
    if os.path.exists(s):
        with open(s, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("endpointer", d)
    return json.loads(s)


def summarize(recs):
    out = {}
    for c in CONDITIONS:
        rs = [r for r in recs if r["condition"] == c]
        if not rs:
            continue
        missed = [r for r in rs if r["vad_start"] is None]
        det = [r for r in rs if r["vad_start"] is not None]
        ends = [(r["vad_end"] if r["vad_end"] is not None else r["audio_duration"]) - r["speech_end"] for r in det]
        late = [e for e in ends if e >= 0]
        early = [e for e in ends if e < 0]
        starts = [r["vad_start"] - r["speech_start"] for r in det]
        out[c] = {
            "n": len(rs),
            "missed_pct": round(100 * len(missed) / len(rs), 2),
            "early_cut_pct": round(100 * len(early) / len(rs), 2),
            "early_cut_n": len(early),
            "early_cut_median_s": round(statistics.median(early), 3) if early else None,
            "end_delay_p50": round(percentile(late, 50), 3) if late else None,
            "end_delay_p90": round(percentile(late, 90), 3) if late else None,
            "end_delay_max": round(max(late), 3) if late else None,
            "start_err_median": round(statistics.median(starts), 3) if starts else None,
            "ep_compute_total_s": round(sum(r["processing_time"] for r in rs), 2),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="default", help="'default', inline JSON, or a config file with an 'endpointer' key")
    ap.add_argument("--manifest", default="data/eval/manifest.jsonl")
    ap.add_argument("--audio-root", default="data/eval")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N utterances per condition")
    ap.add_argument("--out", default=None, help="write per-utterance records (jsonl)")
    ap.add_argument("--latency-config", default=None)
    ap.add_argument("--replay", default=None, help="latency.jsonl of a recorded run whose partials feed system.text")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.latency_config) if args.latency_config else load_config()
    sr = cfg["sample_rate"]
    spec = parse_spec(args.spec)
    utts = list(read_jsonl(args.manifest))
    if args.conditions:
        keep = set(args.conditions.split(","))
        utts = [u for u in utts if u["condition"] in keep]
    if args.limit:
        per = {}
        kept = []
        for u in utts:
            per[u["condition"]] = per.get(u["condition"], 0) + 1
            if per[u["condition"]] <= args.limit:
                kept.append(u)
        utts = kept

    system = Replay(args.replay, sr) if args.replay else Silent()
    ep = build_endpointer(cfg, spec, system)
    recs = []
    t0 = time.time()
    outf = open(args.out, "w", encoding="utf-8") if args.out else None
    for i, u in enumerate(utts):
        a = load_audio(os.path.join(args.audio_root, u["audio"]), sr)
        if args.replay:
            system.load(u["id"])
        r = run_streaming(a, sr, u, system, cfg, ep).to_dict()
        r["condition"] = u["condition"]
        r["noise_type"] = (u.get("noise") or {}).get("type")
        r.pop("partials", None); r.pop("text", None)
        recs.append(r)
        if outf:
            outf.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not args.quiet and (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(utts)} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    if outf:
        outf.close()

    summ = summarize(recs)
    limit = cfg["latency_constraint_p90_sec"]
    print(f"endpointer: {describe_endpointer(spec)}   ({len(utts)} utterances, {time.time() - t0:.0f}s)")
    print(f"{'cond':6s} {'n':>4s} {'miss%':>6s} {'early%':>7s} {'(n)':>4s} {'earlyMed':>8s} {'endP50':>7s} {'endP90':>7s} {'endMax':>7s} {'startErr':>8s} {'budget':>7s}")
    for c, s in summ.items():
        f = lambda v, nd=3: "-" if v is None else f"{v:.{nd}f}"
        budget = None if s["end_delay_p90"] is None else limit - s["end_delay_p90"]
        print(f"{c:6s} {s['n']:4d} {s['missed_pct']:6.1f} {s['early_cut_pct']:7.1f} {s['early_cut_n']:4d} {f(s['early_cut_median_s']):>8s} "
              f"{f(s['end_delay_p50']):>7s} {f(s['end_delay_p90']):>7s} {f(s['end_delay_max']):>7s} {f(s['start_err_median']):>8s} {f(budget):>7s}")
    return summ


if __name__ == "__main__":
    main()
