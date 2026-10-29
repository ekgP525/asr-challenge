"""Secondary analysis of the baseline runs: what drives CER and latency.

    python evaluation/analyze.py [--runs runs] [--manifest data/eval/manifest.jsonl] [--out results/analysis.json]

Per system and condition:
  * CER split by source (zeroth / fleurs)
  * CER on utterances the endpointer cut early vs. the rest (early cut truncates the
    tail, so it inflates CER independently of the recognizer)
  * final-latency decomposition: endpoint delay (vad_end - speech_end) vs. compute after it
  * for streaming systems: revision / premature-partial rates
Writes results/analysis.json and prints a markdown summary.
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.latency import percentile  # noqa: E402
from evaluation.metrics import CERAccumulator  # noqa: E402

CONDS = ["clean", "snr20", "snr10", "snr5", "snr0"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--manifest", default="data/eval/manifest.jsonl")
    ap.add_argument("--systems", default="B1,B2,B3,B4")
    ap.add_argument("--out", default="results/analysis.json")
    a = ap.parse_args()
    man = {u["id"]: u for u in read_jsonl(a.manifest)}
    out = {}
    lines = []
    for sid in a.systems.split(","):
        d = os.path.join(a.runs, sid)
        if not os.path.exists(os.path.join(d, "latency.jsonl")):
            continue
        preds = {p["id"]: p["text"] for p in read_jsonl(os.path.join(d, "predictions.jsonl"))}
        recs = {r["id"]: r for r in read_jsonl(os.path.join(d, "latency.jsonl"))}
        streaming = any(len(r.get("partials", [])) > 1 for r in recs.values())
        sysout = {}
        for c in CONDS:
            ids = [i for i in man if man[i]["condition"] == c and i in preds]
            by_src = defaultdict(CERAccumulator)
            cut, kept = CERAccumulator(), CERAccumulator()
            end_delay, compute, finals = [], [], []
            for i in ids:
                u, r = man[i], recs.get(i)
                by_src[u["source"]].add(u["text"], preds[i])
                if r is None:
                    continue
                vend = r["vad_end"] if r["vad_end"] is not None else r["audio_duration"]
                early = vend < u["speech_end"]
                (cut if early else kept).add(u["text"], preds[i])
                end_delay.append(vend - u["speech_end"])
                compute.append(r["t_final"] - vend)
                finals.append(r["t_final"] - u["speech_end"])
            sysout[c] = {
                "cer_by_source": {s: round(acc.cer, 2) for s, acc in by_src.items()},
                "n_early_cut": cut.n_utts, "cer_early_cut": round(cut.cer, 2) if cut.n_utts else None,
                "cer_not_cut": round(kept.cer, 2) if kept.n_utts else None,
                "endpoint_delay_p50": round(statistics.median(end_delay), 3),
                "endpoint_delay_p90": round(percentile(end_delay, 90), 3),
                "compute_after_endpoint_p50": round(statistics.median(compute), 3),
                "compute_after_endpoint_p90": round(percentile(compute, 90), 3),
                "final_p90": round(percentile(finals, 90), 3),
            }
        out[sid] = {"streaming": streaming, "cond": sysout}
        lines.append(f"\n### {sid} ({'streaming' if streaming else 'offline'})\n")
        lines.append("| cond | CER zeroth | CER fleurs | early-cut n | CER cut | CER not cut | endpoint p50/p90 | compute p50/p90 | final p90 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            s = sysout[c]
            lines.append(f"| {c} | {s['cer_by_source'].get('zeroth')} | {s['cer_by_source'].get('fleurs')} | {s['n_early_cut']} | "
                         f"{s['cer_early_cut']} | {s['cer_not_cut']} | {s['endpoint_delay_p50']}/{s['endpoint_delay_p90']} | "
                         f"{s['compute_after_endpoint_p50']}/{s['compute_after_endpoint_p90']} | {s['final_p90']} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
