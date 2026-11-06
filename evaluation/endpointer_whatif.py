"""Estimate what a different endpointer would do to a recorded system run (no GPU).

    python evaluation/endpointer_whatif.py --ep runs/ep/V2.jsonl --run runs/B4 [--result results/B4.json]

Combines per-utterance endpoint times from an endpointer-only evaluation (endpointer_eval.py --out) with the
recorded run's post-endpoint compute (t_final - vad_end per utterance) and per-utterance CER:
  * final latency  = new vad_end + recorded (t_final - vad_end)            -> p50 / p90 per condition
  * CER            = recorded per-utterance CER, except utterances whose cut status changes:
                     newly uncut -> the condition's mean CER of uncut utterances in the recorded run,
                     newly cut   -> the condition's mean CER of cut utterances
This is an estimate: the ASR's output on an utterance it previously saw truncated is unknown. The
streaming tail compute is a good approximation only if the ASR's partial cadence is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.latency import percentile  # noqa: E402
from evaluation.metrics import CERAccumulator  # noqa: E402
from evaluation.stream_sim import load_config  # noqa: E402

CONDITIONS = ["clean", "snr20", "snr10", "snr5", "snr0"]
NOISY = CONDITIONS[1:]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", required=True, help="endpointer_eval.py --out file")
    ap.add_argument("--run", required=True, help="recorded run dir with latency.jsonl + predictions.jsonl")
    ap.add_argument("--manifest", default="data/eval/manifest.jsonl")
    ap.add_argument("--json", default=None, help="write the estimate here")
    args = ap.parse_args(argv)

    limit = load_config()["latency_constraint_p90_sec"]
    man = {u["id"]: u for u in read_jsonl(args.manifest)}
    rec = {r["id"]: r for r in read_jsonl(os.path.join(args.run, "latency.jsonl"))}
    pred = {p["id"]: p["text"] for p in read_jsonl(os.path.join(args.run, "predictions.jsonl"))}
    ep = {r["id"]: r for r in read_jsonl(args.ep)}

    # per-utterance edits / reference chars (leaderboard CER = sum edits / sum chars)
    edits, chars = {}, {}
    for u in rec:
        acc = CERAccumulator()
        acc.add(man[u]["text"], pred.get(u, ""))
        edits[u], chars[u] = acc.edits, acc.ref_chars
    was_cut = {u: rec[u]["vad_end"] is not None and rec[u]["vad_end"] < man[u]["speech_end"] for u in rec}
    rate_cut, rate_uncut = {}, {}        # edit rate of cut / uncut utterances per condition
    for c in CONDITIONS:
        ids = [u for u in rec if man[u]["condition"] == c]
        a = [u for u in ids if was_cut[u]]
        b = [u for u in ids if not was_cut[u]]
        rate_cut[c] = sum(edits[u] for u in a) / max(1, sum(chars[u] for u in a)) if a else 0.7
        rate_uncut[c] = sum(edits[u] for u in b) / max(1, sum(chars[u] for u in b)) if b else 0.0

    out = {}
    for c in CONDITIONS:
        ids = [u for u in ep if man[u]["condition"] == c and u in rec]
        finals = []
        e_est = e_rec = n_chars = 0.0
        n_cut = n_newcut = n_uncut = 0
        for u in ids:
            e, r = ep[u], rec[u]
            new_end = e["vad_end"] if e["vad_end"] is not None else e["audio_duration"]
            tail = r["t_final"] - (r["vad_end"] if r["vad_end"] is not None else r["audio_duration"])
            finals.append(new_end + tail - man[u]["speech_end"])
            now_cut = e["vad_end"] is not None and e["vad_end"] < man[u]["speech_end"]
            n_cut += now_cut
            n_chars += chars[u]; e_rec += edits[u]
            if now_cut and not was_cut[u]:
                n_newcut += 1; e_est += rate_cut[c] * chars[u]
            elif was_cut[u] and not now_cut:
                n_uncut += 1; e_est += rate_uncut[c] * chars[u]
            else:
                e_est += edits[u]
        out[c] = {"n": len(ids), "p50": round(percentile(finals, 50), 3), "p90": round(percentile(finals, 90), 3),
                  "early_cut_n": n_cut, "newly_cut": n_newcut, "newly_uncut": n_uncut,
                  "cer_est": round(100 * e_est / n_chars, 2),
                  "cer_recorded": round(100 * e_rec / n_chars, 2)}
    noisy_cer = round(statistics.mean(out[c]["cer_est"] for c in NOISY), 2)
    meets = all(out[c]["p90"] <= limit for c in NOISY)
    print(f"what-if: {args.ep} on {args.run}")
    print(f"{'cond':6s} {'finalP50':>8s} {'finalP90':>8s} {'cuts':>5s} {'+cut':>5s} {'-cut':>5s} {'CERest':>7s} {'CERrec':>7s}")
    for c, s in out.items():
        print(f"{c:6s} {s['p50']:8.3f} {s['p90']:8.3f} {s['early_cut_n']:5d} {s['newly_cut']:5d} {s['newly_uncut']:5d} {s['cer_est']:7.2f} {s['cer_recorded']:7.2f}")
    print(f"noisy CER est {noisy_cer}  (recorded {round(statistics.mean(out[c]['cer_recorded'] for c in NOISY), 2)})   "
          f"constraint p90<={limit}: {'OK' if meets else 'FAIL'}  (max noisy p90 {max(out[c]['p90'] for c in NOISY)})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"cond": out, "noisy_cer_est": noisy_cer, "meets": meets}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
