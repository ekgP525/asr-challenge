"""Quantify measurement jitter: compare single-run final latencies (the full-set
protocol, repeats=1) with the median of 3 repeats on a subset.

    python baseline/run_qwen3_asr.py --config baseline/configs/B2.json --manifest data/eval/manifest.jsonl \
        --audio-root data/eval --out-dir runs/jitter_B2 --limit 200 --repeats 3
    python evaluation/jitter_check.py --single runs/B2/latency.jsonl --repeated runs/jitter_B2/latency.jsonl \
        --out results/jitter_B2.json

Reports the median and max absolute difference of final latency (s) over the
common utterances and the p90 of each set, so the page can state how much a
single measurement deviates from the 3-run median.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.latency import final_latency, percentile  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, (l for l in f if l.strip()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", required=True, help="latency.jsonl from the repeats=1 full run")
    ap.add_argument("--repeated", required=True, help="latency.jsonl from the repeats=3 subset run")
    ap.add_argument("--out", default="results/jitter.json")
    a = ap.parse_args()
    one, three = load(a.single), load(a.repeated)
    ids = sorted(set(one) & set(three))
    if not ids:
        sys.exit("no common utterances")
    d = [final_latency(one[i]) - final_latency(three[i]) for i in ids]
    out = {
        "n": len(ids),
        "median_abs_diff_s": round(statistics.median(map(abs, d)), 4),
        "max_abs_diff_s": round(max(map(abs, d)), 4),
        "mean_diff_s": round(statistics.mean(d), 4),
        "p90_single_s": round(percentile([final_latency(one[i]) for i in ids], 90), 4),
        "p90_median3_s": round(percentile([final_latency(three[i]) for i in ids], 90), 4),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
