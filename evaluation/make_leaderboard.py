"""results/*.json -> leaderboard (markdown + json) and latency / Term Recall scatter plot.

Ranking rule
------------
* Primary: average Term Recall over the four noisy conditions (higher is better).
* Only systems with p90 Final Latency <= constraint (1.0 s) enter the main ranking.
  Systems that miss it, or have no Track B measurement, are listed separately.
* Tie-break: average CER over the noisy conditions (lower is better).

The JSON output is what the challenge HTML embeds; the PNG is optional
(requires matplotlib) and is skipped with a warning when it is not installed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.latency import load_config  # noqa: E402


def load_results(results_dir: str) -> list:
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(p) == "leaderboard.json":
            continue
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def rank(results: list, limit: float):
    def key(r):
        a = r["track_a"].get("noisy_avg", {})
        return (-a.get("term_recall", 0.0), a.get("cer", 1e9))

    main = [r for r in results if r.get("meets_latency_constraint") is True]
    other = [r for r in results if r.get("meets_latency_constraint") is not True]
    main.sort(key=key)
    other.sort(key=key)
    return main, other


def row(r: dict, rank_no):
    a = r["track_a"]
    b = r.get("track_b") or {}
    def f(x, nd=2):
        return "-" if x is None else f"{x:.{nd}f}"
    return {
        "rank": rank_no,
        "system_id": r["system_id"],
        "model": r["model"],
        "mode": r["mode"],
        "context_prompt": r["context_prompt"],
        "noisy_term_recall": a.get("noisy_avg", {}).get("term_recall"),
        "noisy_cer": a.get("noisy_avg", {}).get("cer"),
        "clean_cer": a.get("clean", {}).get("cer"),
        "clean_term_recall": a.get("clean", {}).get("term_recall"),
        "per_condition": {c: a[c] for c in ("clean", "snr20", "snr10", "snr5", "snr0") if c in a},
        "final_latency_p50": b.get("final_latency_p50"),
        "final_latency_p90": b.get("final_latency_p90"),
        "first_partial_latency": b.get("first_partial_latency"),
        "rtfx": b.get("rtfx"),
        "meets_latency_constraint": r.get("meets_latency_constraint"),
        "hardware": r.get("hardware"),
        "_fmt": f,
    }


def to_markdown(main_rows, other_rows, limit) -> str:
    hdr = ("| # | System | Model | Mode | Context | Noisy TR (%) ↑ | Noisy CER (%) ↓ | Clean CER (%) | "
           "p50 (s) | p90 (s) ↓ | First-partial (s) | RTFx ↑ |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    def line(x):
        f = x["_fmt"]
        return (f"| {x['rank'] if x['rank'] is not None else '-'} | {x['system_id']} | {x['model']} | {x['mode']} | "
                f"{'yes' if x['context_prompt'] else 'no'} | {f(x['noisy_term_recall'])} | {f(x['noisy_cer'])} | "
                f"{f(x['clean_cer'])} | {f(x['final_latency_p50'], 3)} | {f(x['final_latency_p90'], 3)} | "
                f"{f(x['first_partial_latency'], 3)} | {f(x['rtfx'], 1)} |\n")

    md = f"## Main ranking (p90 Final Latency ≤ {limit:.1f} s)\n\n" + hdr
    md += "".join(line(x) for x in main_rows) if main_rows else "| - | (no system meets the constraint yet) |||||||||||\n"
    if other_rows:
        md += "\n## Outside the latency constraint / no Track B measurement\n\n" + hdr
        md += "".join(line(x) for x in other_rows)
    return md


def plot_scatter(rows, limit, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping scatter plot", file=sys.stderr)
        return
    pts = [r for r in rows if r["final_latency_p90"] is not None and r["noisy_term_recall"] is not None]
    if not pts:
        print("no systems with both latency and accuracy; skipping scatter plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in pts:
        ok = r["meets_latency_constraint"]
        ax.scatter(r["final_latency_p90"], r["noisy_term_recall"], s=70,
                   marker="o" if ok else "x", color="#1f77b4" if ok else "#d62728")
        ax.annotate(r["system_id"], (r["final_latency_p90"], r["noisy_term_recall"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axvline(limit, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("p90 Final Latency (s)")
    ax.set_ylabel("Noisy-avg Term Recall (%)")
    ax.set_title("Accuracy vs. latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-md", default="results/leaderboard.md")
    ap.add_argument("--out-json", default="results/leaderboard.json")
    ap.add_argument("--out-png", default="results/latency_vs_term_recall.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    limit = load_config()["latency_constraint_p90_sec"]
    results = load_results(args.results_dir)
    if not results:
        print(f"no result files in {args.results_dir}", file=sys.stderr)
        return 1
    main_r, other_r = rank(results, limit)
    main_rows = [row(r, i + 1) for i, r in enumerate(main_r)]
    other_rows = [row(r, None) for r in other_r]

    md = to_markdown(main_rows, other_rows, limit)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    strip = lambda x: {k: v for k, v in x.items() if not k.startswith("_")}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"latency_constraint_p90_sec": limit,
                   "main": [strip(x) for x in main_rows],
                   "other": [strip(x) for x in other_rows]}, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"wrote {args.out_md}, {args.out_json}")
    if not args.no_plot:
        plot_scatter(main_rows + other_rows, limit, args.out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
